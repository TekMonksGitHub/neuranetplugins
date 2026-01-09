// Thai Document AI OCR - Full Quality Preprocessing
// by TekMonks Ltd - https://tekmonks.com
"use strict";

const os = require("os");
const fs = require("fs");
const path = require("path");
const sharp = require("sharp");
const crypto = require("crypto");
const mammoth = require("mammoth");
const { Readable } = require("stream");
const { exec } = require("child_process");
const memfs = require(`${CONSTANTS.LIBDIR}/memfs.js`);
const { DocumentProcessorServiceClient } = require("@google-cloud/documentai");

const confPath = path.join(__dirname, "conf", "googlebasedocr.json");
const conf = require(confPath);

const KEYFILE = process.env.GOOGLE_APPLICATION_CREDENTIALS || confPath;
const PROCESSOR = process.env.DOC_AI_PROCESSOR || conf.processor;
const DPI = Number(process.env.DPI || conf.dpi);
const SHARP = process.env.SHARP || conf.sharp;
const MEDIAN = Number(process.env.MEDIAN || conf.median);
const UNGAMMA = process.env.UNGAMMA || conf.ungamma;
const MAX_RETRIES = conf.max_retries;
const TARGET_WIDTH = Number(process.env.TARGET_WIDTH || conf.target_width);
const RETRY_DELAY_MS = conf.retry_delay_ms;
const MAX_FILE_SIZE_MB = conf.max_file_size;

// Concurrency limits for pipeline processing (optimized for 6GB RAM servers)
const CONCURRENCY_IMAGE_ENHANCEMENT = Number(process.env.CONCURRENCY_IMAGE_ENHANCEMENT || conf.concurrency_image_enhancement);
const CONCURRENCY_GOOGLE_OCR = Number(process.env.CONCURRENCY_GOOGLE_OCR || conf.concurrency_google_ocr);
const CONCURRENCY_PDF_CONVERSION = Number(process.env.CONCURRENCY_PDF_CONVERSION || conf.concurrency_pdf_conversion);

/* ======================================================
   C — MemFS Helper Wrappers (preserve names you like)
   ====================================================== */
// memfs convenience wrappers
const mexists = async (p) => memfs.access(p).then(() => true).catch(() => false);
const mread = async (p) => memfs.readFile(p);
const mwrite = async (p, d) => memfs.writeFile(p, d);
const mstat = async (p) => memfs.stat(p);
const mdir = async (p) => (memfs.readDir || memfs.readdir).call(memfs, p);
const mrm = async (p, opts) => memfs.rm(p, opts);
const maccess = async (p, mode) => memfs.access(p, mode);
const mmkdir = async (p, opts) => memfs.mkdir(p, opts);
const munlink = async (p) => memfs.unlink(p);

/* ======================================================
   Concurrency Control Utility
   - Limits the number of concurrent async operations
   - Prevents memory spikes by controlling parallel tasks
   - No external dependencies (p-limit alternative)
   ====================================================== */
function createConcurrencyLimiter(maxConcurrent) {
    let activeCount = 0;
    const queue = [];

    /**
     * Wraps an async function to run with concurrency limit
     * @param {Function} fn - Async function to execute
     * @returns {Promise} - Promise that resolves when fn completes
     */
    const run = async (fn) => {
        // Wait if we're at max concurrency
        while (activeCount >= maxConcurrent) {
            await new Promise(resolve => queue.push(resolve));
        }

        activeCount++;
        try {
            return await fn();
        } finally {
            activeCount--;
            // Allow next queued operation to proceed
            if (queue.length > 0) {
                const resolve = queue.shift();
                resolve();
            }
        }
    };

    return run;
}

/* ======================================================
   Initialize Document AI Client (top-level await as before)
   ====================================================== */
let documentAiClient;
(async () => {
    try {
        if (!await mexists(KEYFILE)) throw new Error(`Credentials file not found: ${KEYFILE}`);
        documentAiClient = new DocumentProcessorServiceClient({ keyFilename: KEYFILE });
    } catch (initError) {
        console.error(`Failed to initialize Document AI client: ${initError.message}`);
        process.exit(1);
    }
})();


const execp = (commandString, description = "Command") =>
    new Promise((resolve, reject) => {
        exec(commandString, { maxBuffer: 1024 * 1024 * 400 }, (err, stdout, stderr) => {
            if (err) {
                const errorDetails = [
                    `${description} failed`,
                    stderr ? `Error: ${stderr.trim()}` : err.message,
                    err.code ? `Exit code: ${err.code}` : null
                ].filter(Boolean).join(' | ');
                return reject(new Error(errorDetails));
            }
            resolve((stdout || "").trim());
        });
    });

/* ======================================================
   Utility functions (renamed for clarity)
   ====================================================== */
async function _ensureDirectoryExists(directoryPath) {
    try { await maccess(directoryPath, fs.constants.F_OK); }
    catch (err) { await mmkdir(directoryPath, { recursive: true }); }
    try { await maccess(directoryPath, fs.constants.W_OK); }
    catch (err) {
        if (err.code === 'EACCES') throw new Error(`Permission denied: Cannot create/write to directory: ${directoryPath}\nTry: sudo chmod 755 ${directoryPath}`);
        else if (err.code === 'ENOSPC') throw new Error(`Disk full: Cannot create directory: ${directoryPath}\nFree up some disk space.`);
        else throw new Error(`Failed to create directory ${directoryPath}: ${err.message}`);
    }
}

function _sortFilesByPageNumber(a, b) {
    try {
        const matchA = a.match(/(\d+)(?=\D*$)/);
        const matchB = b.match(/(\d+)(?=\D*$)/);
        const na = matchA ? Number(matchA[1]) : 0;
        const nb = matchB ? Number(matchB[1]) : 0;
        return na - nb;
    } catch (err) {
        LOG.info(`File sorting failed: ${err.message}`);
        return 0;
    }
}

function _normalizeExtractedText(text) {
    return (text || "")
        .replace(/\r\n/g, "\n")
        .replace(/[ \t]+\n/g, "\n")
        .replace(/\n{3,}/g, "\n\n")
        .trim();
}

/* ======================================================
   Validation helpers (renamed)
   ====================================================== */
function _validateBase64PdfString(base64String) {
    if (!base64String || typeof base64String !== 'string') throw new Error('Invalid input: base64 string is required');
    if (base64String.trim().length === 0) throw new Error('Invalid input: base64 string is empty');
    let cleanBase64String = base64String.trim();
    if (cleanBase64String.includes('base64,')) cleanBase64String = cleanBase64String.split('base64,')[1];
    const base64Regex = /^[A-Za-z0-9+/]+={0,2}$/;
    if (!base64Regex.test(cleanBase64String)) throw new Error('Invalid base64 format. Contains invalid characters');
    if (cleanBase64String.length < 100) throw new Error('Invalid PDF: base64 string too short (possibly corrupted)');
    return cleanBase64String;
}

function _decodeBase64PdfToBuffer(cleanBase64String) {
    let pdfFileBuffer;
    try {
        pdfFileBuffer = Buffer.from(cleanBase64String, 'base64');
        const pdfHeader = pdfFileBuffer.toString('ascii', 0, 5);
        if (!pdfHeader.startsWith('%PDF')) throw new Error('Invalid PDF: file signature not found. May be corrupted or not a PDF');
    } catch (err) {
        if (err.message.includes('Invalid PDF')) throw err;
        throw new Error(`Failed to decode base64: ${err.message}`);
    }
    const fileSizeMB = pdfFileBuffer.length / (1024 * 1024);
    if (fileSizeMB > MAX_FILE_SIZE_MB) throw new Error(`PDF too large (${fileSizeMB.toFixed(2)} MB). Max: ${MAX_FILE_SIZE_MB} MB`);
    return pdfFileBuffer;
}

async function _validateGoogleCredentialsFile() {
    try {
        const credBuffer = await mread(KEYFILE);
        const credData = JSON.parse(credBuffer.toString('utf8'));
        if (!credData.type || !credData.project_id) throw new Error(`Invalid credentials file format: ${KEYFILE}`);
    } catch (err) {
        throw new Error(`Cannot read credentials file: ${err.message}`);
    }
}

/* ======================================================
   External-tool helpers
   ====================================================== */

/**
 * Get the number of pages in a PDF file using mutool
 * @param {string} pdfPath - Path to PDF file
 * @returns {Promise<number>} - Number of pages
 */
async function _getPdfPageCount(pdfPath) {
    try {
        // mutool show <file.pdf> trailer/Root/Pages/Count
        // Returns just the page count number
        const output = await execp(`mutool show "${pdfPath}" trailer/Root/Pages/Count`, 'Get PDF page count');
        const pageCount = parseInt(output.trim());
        if (isNaN(pageCount) || pageCount < 1) throw new Error(`Invalid page count: ${output}`);
        return pageCount;
    } catch (err) {
        const errorMsg = err.message.toLowerCase();
        if (errorMsg.includes('command not found')) {
            throw new Error(`mutool not installed. Install: sudo apt-get install mupdf-tools`);
        }
        if (errorMsg.includes('encrypted') || errorMsg.includes('password')) {
            throw new Error(`PDF is password protected: ${pdfPath}`);
        }
        throw new Error(`Failed to get PDF page count: ${err.message}`);
    }
}

/**
 * Convert PDF to PNG images using mutool with parallel processing
 * - Uses mutool (faster than pdftoppm, 2-3x speedup)
 * - Converts pages in parallel with concurrency control
 * - Maintains sequential page ordering
 *
 * @param {string} temporaryPdfFilePath - Path to PDF file
 * @param {string} outputPagesDirectory - Directory to save PNG files
 * @param {number} dpi - DPI for image resolution
 * @returns {Promise<string[]>} - Array of PNG file paths in page order
 */
async function _convertPdfToPngImages(temporaryPdfFilePath, outputPagesDirectory, dpi = DPI) {
    try {
        LOG.info(`Converting PDF to PNG @ ${dpi} DPI using mutool (parallel)...`);
        await _ensureDirectoryExists(outputPagesDirectory);

        // Get total page count
        const pageCount = await _getPdfPageCount(temporaryPdfFilePath);
        LOG.info(`PDF has ${pageCount} page(s)`);

        const prefix = path.join(outputPagesDirectory, "page");

        // Create concurrency limiter for PDF conversion
        const conversionLimiter = createConcurrencyLimiter(CONCURRENCY_PDF_CONVERSION);

        // Convert pages in parallel
        LOG.info(`Converting pages with concurrency limit: ${CONCURRENCY_PDF_CONVERSION}`);
        await Promise.all(
            Array.from({ length: pageCount }, (_, i) => i + 1).map(pageNum =>
                conversionLimiter(async () => {
                    try {
                        // mutool draw -o <output> -r <dpi> -F png <pdf> <pageNum>
                        // Note: mutool uses 1-based page numbering
                        const outputFile = `${prefix}-${pageNum}.png`;
                        await execp(`mutool draw -o "${outputFile}" -r ${dpi} -F png "${temporaryPdfFilePath}" ${pageNum}`,
                            `Convert page ${pageNum}` );
                        LOG.info(`  Page ${pageNum}/${pageCount} converted`);
                    } catch (err) { throw new Error(`Page ${pageNum} conversion failed: ${err.message}`); } 
             })
            )
        );

        // Read and sort generated files
        let fileList; try {
            fileList = (await mdir(outputPagesDirectory))
                .filter((file) => file.startsWith("page") && file.endsWith(".png"))
                .sort(_sortFilesByPageNumber)
                .map((file) => path.join(outputPagesDirectory, file));
        } catch (err) { throw new Error(`Cannot read output directory ${outputPagesDirectory}: ${err.message}`); }

        if (!fileList.length) throw new Error(`No PNG files generated. PDF may be encrypted, empty, or corrupted: ${temporaryPdfFilePath}`);

        // Validate all pages were converted
        if (fileList.length !== pageCount) throw new Error(`Expected ${pageCount} pages but got ${fileList.length} PNG files`);

        // Calculate total size
        let totalSize = 0; for (const pageImagePath of fileList) {
            try {
                const stats = await mstat(pageImagePath);
                totalSize += stats.size;
                if (stats.size < 1000) {
                    LOG.info(`Warning: ${path.basename(pageImagePath)} is very small (${stats.size} bytes)`);
                }
            } catch (err) { LOG.info(`Cannot stat file ${path.basename(pageImagePath)}: ${err.message}`); }
        }

        LOG.info(`Generated ${fileList.length} page(s) | Total: ${(totalSize / 1024 / 1024).toFixed(2)} MB`);
        return fileList;
    } catch (err) {
        const errorMsg = err.message.toLowerCase();
        // Provide helpful error messages
        if (errorMsg.includes('command not found')) throw new Error(`mutool not installed. Install: sudo apt-get install mupdf-tools`);
        if (errorMsg.includes('syntax error') || errorMsg.includes('damaged')) throw new Error(`PDF file is corrupted or invalid: ${temporaryPdfFilePath}`);
        if (errorMsg.includes('encrypted') || errorMsg.includes('password')) throw new Error(`PDF is password protected: ${temporaryPdfFilePath}`);
        if (errorMsg.includes('permission denied')) throw new Error(`Cannot read PDF file: ${temporaryPdfFilePath}`);
        if (errorMsg.includes('no space left') || errorMsg.includes('enospc')) throw new Error(`Disk full! Cannot create PNG files in: ${outputPagesDirectory}`);
        throw new Error(`convertPdfToPngImages failed: ${err.message}`);
    }
}

/**
 * Enhance image using Sharp library (replaces ImageMagick)
 * - Uses Sharp for in-process image manipulation (10x lower memory than ImageMagick)
 * - No external process spawning (faster and more reliable)
 * - Better memory control for 6GB RAM servers
 *
 * Pipeline steps:
 * 1. Read image and convert to grayscale
 * 2. Resize to target width (maintains aspect ratio)
 * 3. Apply contrast stretching
 * 4. Apply median filter (noise reduction)
 * 5. Apply unsharp mask (sharpening)
 * 6. Normalize and threshold for OCR
 *
 * @param {string} inputImagePath - Path to input PNG image
 * @param {string} outputImagePath - Path to save enhanced image
 * @throws {Error} If image processing fails
 */
async function _enhanceImageUsingSharp(inputImagePath, outputImagePath) {
    try {
        // Validate input file exists and is readable
        try { await maccess(inputImagePath, fs.constants.R_OK); }
        catch (err) { throw new Error(`Input image not found: ${inputImagePath}`); }

        const inputStats = await mstat(inputImagePath);
        if (inputStats.size === 0) throw new Error(`Input image is empty: ${inputImagePath}`);

        // Read the input image from memfs
        const inputBuffer = await mread(inputImagePath);

        // Parse SHARP config value: "0x1.0" -> sigma=1.0
        const sharpSigma = parseFloat(SHARP.split('x')[1]) || 1.0;

        // Parse CONTRAST_STRETCH: "0.5%x0.5%" -> we'll use normalize for similar effect

        // Create Sharp pipeline
        let pipeline = sharp(inputBuffer, {
            // Limit resources to prevent memory issues on 6GB servers
            limitInputPixels: 268402689, // ~16k x 16k max
            sequentialRead: true
        });

        // Step 1: Convert to grayscale and resize
        pipeline = pipeline
            .grayscale()
            .resize(TARGET_WIDTH, null, {
                fit: 'inside',
                kernel: 'lanczos3', // High-quality downsampling (equivalent to ImageMagick's Lanczos filter)
                withoutEnlargement: true
            });

        // Step 2: Normalize (auto-levels, similar to contrast-stretch)
        pipeline = pipeline.normalize();

        // Step 3: Apply median filter for noise reduction (if configured)
        // Note: Sharp doesn't have built-in median filter, but we can use blur as alternative
        if (MEDIAN > 0) {
            pipeline = pipeline.median(MEDIAN);
        }

        // Step 4: Sharpen the image (unsharp mask)
        // Sharp's sharpen: sigma ~= radius, higher values = more sharpening
        if (sharpSigma > 0) {
            pipeline = pipeline.sharpen({
                sigma: sharpSigma,
                m1: 1.0,  // Sharpening amount
                m2: 0.5   // Edge threshold
            });
        }

        // Step 5: Apply gamma correction (ungamma)
        const gammaValue = parseFloat(UNGAMMA) || 1.0;
        if (gammaValue !== 1.0) {
            pipeline = pipeline.gamma(gammaValue);
        }

        // Step 6: Threshold for binary (black/white) image - optimal for OCR
        // We'll use a combination of techniques to simulate ImageMagick's adaptive-threshold
        pipeline = pipeline
            .linear(1.2, -(128 * 0.2)) // Increase contrast before thresholding
            .threshold(128, { grayscale: false }); // Binary threshold

        // Output as PNG
        pipeline = pipeline.png({
            compressionLevel: 6, // Balance between speed and file size
            adaptiveFiltering: false
        });

        // Execute the pipeline and write to memfs
        const outputBuffer = await pipeline.toBuffer();
        await mwrite(outputImagePath, outputBuffer);

        // Validate output
        try { await maccess(outputImagePath, fs.constants.F_OK); }
        catch (err) { throw new Error(`Output file not created: ${outputImagePath}`); }

        const outputStats = await mstat(outputImagePath);
        if (outputStats.size === 0) throw new Error(`Output file is empty: ${outputImagePath}`);

    } catch (err) {
        const errorMsg = err.message || '';

        // Handle Sharp-specific errors
        if (errorMsg.includes('Input buffer') || errorMsg.includes('unsupported')) {
            throw new Error(`Invalid or corrupted image: ${inputImagePath}`);
        }
        if (errorMsg.includes('memory') || errorMsg.includes('allocation')) {
            throw new Error(`Sharp memory limit exceeded. Image too large (${(inputStats.size / 1024 / 1024).toFixed(2)} MB)`);
        }

        throw new Error(`Sharp image enhancement failed: ${err.message}`);
    }
}

async function _performDocumentAiOcrOnImage(imagePath, retryCount = 0) {
    try {
        try { await maccess(imagePath, fs.constants.R_OK); }
        catch (err) { throw new Error(`Image not found: ${imagePath}`); }
        const stats = await mstat(imagePath);
        const fileSizeMB = stats.size / 1024 / 1024;
        if (fileSizeMB > MAX_FILE_SIZE_MB) throw new Error(`Image too large (${fileSizeMB.toFixed(2)} MB). Max: ${MAX_FILE_SIZE_MB} MB`);
        const bytes = await mread(imagePath);
        const request = {
            name: PROCESSOR,
            rawDocument: { content: bytes.toString("base64"), mimeType: "image/png" }
        };
        const [documentAiResult] = await documentAiClient.processDocument(request);
        if (!documentAiResult || !documentAiResult.document) throw new Error("Document AI returned empty response");
        return documentAiResult.document;
    } catch (apiErr) {
        const errorMsg = apiErr.message || '';
        const errorCode = apiErr.code || '';
        if (errorCode === 7 || errorMsg.includes('authentication')) throw new Error(`Authentication failed. Check credentials file: ${KEYFILE}`);
        if (errorCode === 8 || errorCode === 429 || errorMsg.includes('quota') || errorMsg.includes('rate limit')) {
            if (retryCount < MAX_RETRIES) {
                const waitTime = RETRY_DELAY_MS * Math.pow(2, retryCount);
                LOG.info(`Rate limited. Retrying in ${waitTime / 1000}s... (${retryCount + 1}/${MAX_RETRIES})`);
                await new Promise(resolve => setTimeout(resolve, waitTime));
                return _performDocumentAiOcrOnImage(imagePath, retryCount + 1);
            }
            throw new Error(`Rate limit exceeded after ${MAX_RETRIES} retries`);
        }
        if (errorMsg.includes('ECONNREFUSED') || errorMsg.includes('ETIMEDOUT') || errorMsg.includes('network')) {
            if (retryCount < MAX_RETRIES) {
                LOG.info(`Network error. Retrying... (${retryCount + 1}/${MAX_RETRIES})`);
                await new Promise(resolve => setTimeout(resolve, RETRY_DELAY_MS));
                return _performDocumentAiOcrOnImage(imagePath, retryCount + 1);
            }
            throw new Error(`Network error after ${MAX_RETRIES} retries`);
        }
        if (errorCode === 5 || errorMsg.includes('not found')) throw new Error(`Invalid processor ID: ${PROCESSOR}`);
        throw new Error(`Document AI failed: ${errorMsg}`);
    }
}

/* ======================================================
   Core Pipeline with Parallel Processing
   ====================================================== */

/**
 * Main OCR Processing Pipeline with Full Parallel Execution
 *
 * ARCHITECTURE:
 * -------------
 * Traditional Sequential Flow (OLD):
 *   For each page: PDF→PNG → Enhance → OCR (wait) → next page
 *   Total time: N pages × (T1 + T2 + T3) = ~90 seconds for 10 pages
 *
 * New Pipeline Flow (FULLY OPTIMIZED):
 *   Stage 1: PDF → PNGs (parallel with limit of 4, using mutool)
 *   Stage 2: PNG → Enhanced PNG (parallel with limit of 2, true pipeline)
 *   Stage 3: Enhanced PNG → OCR (parallel with limit of 3, true pipeline)
 *
 *   Timeline visualization for 3 pages:
 *   Time → 0s    2s    4s    6s    8s    10s   12s   14s   16s
 *   Page 1: [PDF→PNG] → [Enhance] ────→ [OCR──────────────]
 *   Page 2: [PDF→PNG] → [Enhance] ────→ [OCR──────────────]
 *   Page 3:      [PDF→PNG] → [Enhance] → [OCR──────────────]
 *
 *   Key Features:
 *   - PDF→PNG: Parallel conversion using mutool (2-3x faster than pdftoppm)
 *   - Enhancement: Starts as soon as PNG is ready (producer-consumer pattern)
 *   - OCR: Starts as soon as enhancement completes (true pipeline)
 *
 *   Total time: ~25-30 seconds for 10 pages
 *   Speedup: ~3-4x faster than sequential processing
 *
 * CONCURRENCY CONTROL:
 * -------------------
 * - PDF Conversion (mutool): Max 4 concurrent (balance CPU/disk I/O)
 * - Image Enhancement (Sharp): Max 2 concurrent (prevents memory spikes on 6GB RAM)
 * - Google OCR API: Max 3 concurrent (respects rate limits, good parallelism)
 *
 * MEMORY USAGE (6GB Server):
 * --------------------------
 * - 4 × mutool processes: ~400-800MB
 * - 2 × Sharp operations: ~200-400MB
 * - 3 × OCR API calls: ~150-300MB
 * - Peak total: ~1.5GB (safe for 6GB servers)
 *
 * @param {Buffer} pdfFileBuffer - PDF file as Buffer
 * @param {boolean} includePageMarkers - Whether to add "===== PAGE N =====" markers
 * @returns {Promise<Object>} - { success, text, pages, length }
 */
async function _runOcrProcessingPipeline(pdfFileBuffer, includePageMarkers = true) {
    const sessionId = crypto.randomBytes(8).toString('hex');
    const temporarySessionDirectory = path.join(os.tmpdir(), `ocr-${sessionId}`);
    const temporaryPdfFilePath = path.join(temporarySessionDirectory, "input.pdf");
    const temporaryPagesDirectory = path.join(temporarySessionDirectory, "pages");
    const temporaryEnhancedImagesDirectory = path.join(temporarySessionDirectory, "ocr_best");

    try {
        await _ensureDirectoryExists(temporarySessionDirectory);
        await _ensureDirectoryExists(temporaryPagesDirectory);
        await _ensureDirectoryExists(temporaryEnhancedImagesDirectory);

        await mwrite(temporaryPdfFilePath, pdfFileBuffer);
        LOG.info("Temporary PDF created\n");

        // ============================================================
        // STAGE 1: Convert PDF to PNG images (parallel using mutool)
        // ============================================================
        const pageImageList = await _convertPdfToPngImages(temporaryPdfFilePath, temporaryPagesDirectory, DPI);
        const totalPages = pageImageList.length;

        // ============================================================
        // STAGE 2 & 3: TRUE PIPELINE Processing (Producer-Consumer)
        // ============================================================
        LOG.info("\nStarting true pipeline processing...");
        LOG.info(`Concurrency limits: Enhancement=${CONCURRENCY_IMAGE_ENHANCEMENT}, OCR=${CONCURRENCY_GOOGLE_OCR}\n`);

        // Create concurrency limiters
        const enhancementLimiter = createConcurrencyLimiter(CONCURRENCY_IMAGE_ENHANCEMENT);
        const ocrLimiter = createConcurrencyLimiter(CONCURRENCY_GOOGLE_OCR);

        // Create array to store results in correct page order
        const pageResults = new Array(totalPages);

        // Enhancement queue: Stores enhanced image paths ready for OCR
        const enhancedQueue = [];
        const enhancedQueueResolvers = [];

        /**
         * PRODUCER: Enhancement Stage
         * - Enhances images with concurrency limit
         * - Pushes enhanced images to queue for OCR
         * - OCR can start immediately when images are ready
         */
        const enhancementPromises = pageImageList.map(async (pageImagePath, pageIndex) => {
            const pageNumber = pageIndex + 1;
            const base = path.basename(pageImagePath).replace(/\.png$/i, "");
            const enhancedImageOutputPath = path.join(temporaryEnhancedImagesDirectory, `${base}-ocr-best.png`);

            try {
                LOG.info(`  [Page ${pageNumber}/${totalPages}] Enhancing ${path.basename(pageImagePath)}`);

                // Enhance image with concurrency control
                await enhancementLimiter(async () => {
                    await _enhanceImageUsingSharp(pageImagePath, enhancedImageOutputPath);
                });

                LOG.info(`  [Page ${pageNumber}/${totalPages}] Enhancement complete → Queuing for OCR`);

                // Add enhanced image to queue for OCR processing
                const queueItem = { enhancedImageOutputPath, pageIndex, pageNumber };
                enhancedQueue.push(queueItem);

                // Notify waiting OCR consumers that new work is available
                if (enhancedQueueResolvers.length > 0) {
                    const resolver = enhancedQueueResolvers.shift();
                    resolver(queueItem);
                }
            } catch (err) {
                throw new Error(`Enhancement failed for page ${pageNumber}: ${err.message}`);
            }
        });

        /**
         * CONSUMER: OCR Stage
         * - Processes enhanced images as soon as they're available
         * - Doesn't wait for all enhancements to complete
         * - True pipeline: Page 2 OCR can run while Page 3 is being enhanced
         */
        const ocrPromises = pageImageList.map(async (_, pageIndex) => {
            try {
                // Wait for enhanced image to be available in queue
                const getNextEnhancedImage = async () => {
                    // Check if image already in queue
                    const existingItem = enhancedQueue.find(item => item.pageIndex === pageIndex);
                    if (existingItem) {
                        // Remove from queue and return
                        const idx = enhancedQueue.indexOf(existingItem);
                        enhancedQueue.splice(idx, 1);
                        return existingItem;
                    }

                    // Wait for producer to add this page to queue
                    return new Promise(resolve => {
                        const checkQueue = () => {
                            const item = enhancedQueue.find(i => i.pageIndex === pageIndex);
                            if (item) {
                                const idx = enhancedQueue.indexOf(item);
                                enhancedQueue.splice(idx, 1);
                                resolve(item);
                            } else {
                                enhancedQueueResolvers.push(resolve);
                            }
                        };
                        checkQueue();
                    });
                };

                const { enhancedImageOutputPath, pageNumber } = await getNextEnhancedImage();

                // Perform OCR with concurrency control
                LOG.info(`  [Page ${pageNumber}/${totalPages}] Starting OCR`);
                const ocrResult = await ocrLimiter(async () => {
                    return await _performDocumentAiOcrOnImage(enhancedImageOutputPath);
                });
                LOG.info(`  [Page ${pageNumber}/${totalPages}] OCR complete ✓`);

                // Store result in correct order
                const pageExtractedText = _normalizeExtractedText(ocrResult?.text || "");
                pageResults[pageIndex] = pageExtractedText;

            } catch (err) {
                throw new Error(`OCR processing failed for page ${pageIndex + 1}: ${err.message}`);
            }
        });

        // Wait for both enhancement and OCR pipelines to complete
        await Promise.all([...enhancementPromises, ...ocrPromises]);

        LOG.info("\nAll pages processed (true pipeline) ✓\n");

        // ============================================================
        // Combine results in correct page order
        // ============================================================
        let fullExtractedText = "";
        for (let index = 0; index < pageResults.length; index++) {
            const pageExtractedText = pageResults[index];
            if (includePageMarkers) {
                fullExtractedText += `\n\n===== PAGE ${index + 1} =====\n\n${pageExtractedText}\n`;
            } else {
                if (index > 0 && pageExtractedText) fullExtractedText += "\n\n";
                fullExtractedText += pageExtractedText;
            }
        }

        LOG.info("=".repeat(80));
        LOG.info("EXTRACTED TEXT:");
        LOG.info("=".repeat(80));
        LOG.info(fullExtractedText.trim());
        LOG.info("=".repeat(80));
        LOG.info(`\nTotal pages: ${pageResults.length}`);
        LOG.info(`Text length: ${fullExtractedText.trim().length} characters\n`);

        await mrm(temporarySessionDirectory, { recursive: true, force: true });
        LOG.info("Temporary files cleaned up\n");

        return {
            success: true,
            text: fullExtractedText.trim(),
            pages: pageResults.length,
            length: fullExtractedText.trim().length
        };
    } catch (err) {
        LOG.error("\nERROR:", err.message);
        await mrm(temporarySessionDirectory, { recursive: true, force: true });
        throw err;
    }
}

/* ======================================================
   Public API (renamed)
   ====================================================== */
async function _processPdfBase64WithPageMarkers(base64PdfString) {
    LOG.info("Starting Thai OCR Pipeline (Base64 mode with full preprocessing)...\n");
    try {
        const cleanBase64String = _validateBase64PdfString(base64PdfString);
        const pdfFileBuffer = _decodeBase64PdfToBuffer(cleanBase64String);
        await _validateGoogleCredentialsFile();
        LOG.info("Input validation passed\n");
        return await _runOcrProcessingPipeline(pdfFileBuffer, true);
    } catch (err) {
        LOG.error("\nERROR:", err.message);
        throw err;
    }
}

async function _processPdfFileWithPageMarkers(filePath) {
    LOG.info("Starting Thai OCR Pipeline (File mode with full preprocessing)...\n");
    try {
        LOG.info("Validating file...");
        if (!await mexists(filePath)) throw new Error(`File not found: ${filePath}`);
        try { await maccess(filePath, fs.constants.R_OK); }
        catch (err) { throw new Error(`Cannot read file: ${filePath}. Permission denied`); }
        const fileStats = await mstat(filePath);
        if (fileStats.size === 0) throw new Error(`File is empty (0 bytes): ${filePath}`);
        const fileSizeMB = fileStats.size / (1024 * 1024);
        if (fileSizeMB > MAX_FILE_SIZE_MB) throw new Error(`File too large (${fileSizeMB.toFixed(2)} MB). Max: ${MAX_FILE_SIZE_MB} MB`);
        LOG.info(`File size: ${(fileStats.size / 1024).toFixed(2)} KB`);
        const ext = path.extname(filePath).toLowerCase();
        if (ext !== '.pdf') LOG.info(`Warning: File extension is '${ext}', expected '.pdf'`);
        let pdfFileBuffer;
        try { pdfFileBuffer = await mread(filePath); }
        catch (err) { throw new Error(`Failed to read file: ${err.message}`); }
        const pdfHeader = pdfFileBuffer.toString('ascii', 0, 5);
        if (!pdfHeader.startsWith('%PDF')) throw new Error(`Invalid PDF file: signature not found. File may be corrupted or not a PDF`);
        LOG.info("File validation passed\n");
        return await _runOcrProcessingPipeline(pdfFileBuffer, true);
    } catch (err) {
        LOG.error("\nERROR:", err.message);
        throw err;
    }
}

async function processPdfBase64WithoutPageMarkers(base64PdfString) {
    LOG.info("Starting Thai OCR Pipeline (With preprocessing - no page markers)...\n");
    try {
        const cleanBase64String = _validateBase64PdfString(base64PdfString);
        const pdfFileBuffer = _decodeBase64PdfToBuffer(cleanBase64String);
        await _validateGoogleCredentialsFile();
        LOG.info("Input validation passed\n");
        return await _runOcrProcessingPipeline(pdfFileBuffer, false);
    } catch (err) {
        LOG.error("\nERROR:", err.message);
        throw err;
    }
}

async function _processPdfFileWithoutPageMarkers(filePath) {
    LOG.info("Starting Thai OCR Pipeline (File mode with preprocessing - no page markers)...\n");
    try {
        LOG.info("Validating file...");
        if (!await mexists(filePath)) throw new Error(`File not found: ${filePath}`);
        try { await maccess(filePath, fs.constants.R_OK); }
        catch (err) { throw new Error(`Cannot read file: ${filePath}. Permission denied`); }
        const fileStats = await mstat(filePath);
        if (fileStats.size === 0) throw new Error(`File is empty (0 bytes): ${filePath}`);
        const fileSizeMB = fileStats.size / (1024 * 1024);
        if (fileSizeMB > MAX_FILE_SIZE_MB) throw new Error(`File too large (${fileSizeMB.toFixed(2)} MB). Max: ${MAX_FILE_SIZE_MB} MB`);
        LOG.info(`File size: ${(fileStats.size / 1024).toFixed(2)} KB`);
        const ext = path.extname(filePath).toLowerCase();
        if (ext !== '.pdf') LOG.info(`Warning: File extension is '${ext}', expected '.pdf'`);
        let pdfFileBuffer;
        try { pdfFileBuffer = await mread(filePath); }
        catch (err) { throw new Error(`Failed to read file: ${err.message}`); }
        const pdfHeader = pdfFileBuffer.toString('ascii', 0, 5);
        if (!pdfHeader.startsWith('%PDF')) throw new Error(`Invalid PDF file: signature not found. File may be corrupted or not a PDF`);
        LOG.info("File validation passed\n");
        return await _runOcrProcessingPipeline(pdfFileBuffer, false);
    } catch (err) {
        LOG.error("\nERROR:", err.message);
        throw err;
    }
}

/* ======================================================
   CLI behavior (preserve original semantics)
   ====================================================== */
if (require.main === module) {
    const cliInputArgument = process.argv[2];
    const markersEnabled = process.argv[3];
    if (!cliInputArgument) {
        LOG.error("Usage:");
        LOG.error("  With file: node ocr.js /path/to/file.pdf");
        LOG.error("  With base64: node ocr.js <base64-string> [markers_enables]");
        process.exit(1);
    }
    (async () => {
        try {
            if (!markersEnabled) {
                if (await mexists(cliInputArgument)) await _processPdfFileWithoutPageMarkers(cliInputArgument);
                else await processPdfBase64WithoutPageMarkers(cliInputArgument);
                process.exit(0);
            }
            if (await mexists(cliInputArgument)) await _processPdfFileWithPageMarkers(cliInputArgument);
            else await _processPdfBase64WithPageMarkers(cliInputArgument);
            process.exit(0);
        } catch (_) { process.exit(1); }
    })();
}

/* ======================================================
   DOCX Native Text Extraction (No OCR)
   ====================================================== */

/**
 * Extract raw text from DOCX buffer using Mammoth
 * - Preserves Thai characters
 * - Fast and memory-safe
 * - No OCR involved
 *
 * @param {Buffer} docxBuffer
 * @returns {Promise<Buffer>}
 */
async function _extractDocxText(docxBuffer) {
    try {
        if (!docxBuffer || !Buffer.isBuffer(docxBuffer)) {
            throw new Error("Invalid DOCX buffer");
        }

        const result = await mammoth.extractRawText({ buffer: docxBuffer });

        const text = (result.value || "")
            .replace(/\r\n/g, "\n")
            .replace(/\n{3,}/g, "\n\n")
            .trim();

        return Buffer.from(text, "utf8");
    } catch (err) {
        throw new Error(`DOCX extraction failed: ${err.message}`);
    }
}

/* ======================================================
   Stream-based OCR Functions
   ====================================================== */

/** Convert a readable stream to a single Buffer */
async function _streamToBuffer(readableStream) {
    return new Promise((resolve, reject) => {
        const chunks = [];
        readableStream.on("data", chunk => chunks.push(chunk));
        readableStream.on("end", () => resolve(Buffer.concat(chunks)));
        readableStream.on("error", reject);
    });
}

async function _streamToBase64(readableStream) {
    return new Promise((resolve, reject) => {
        const chunks = [];
        readableStream.on("data", chunk => chunks.push(chunk));
        readableStream.on("end", () => {
            const buffer = Buffer.concat(chunks);
            resolve(buffer.toString("base64"));
        });
        readableStream.on("error", reject);
    });
}

/** Extract text buffer from a PDF stream and return full text buffer */
async function getContent(readableStream, fileName) {
    try {
        if (!readableStream) throw new Error("getContent: readableStream is required");
        const ext = path.extname(fileName).toLowerCase();
        // 1️. Direct text-based formats (no OCR)
        const directExtensions = [".txt", ".md", ".html", ".htm", ".csv", ".json", ".xml", ".log", ".rtf", ".tex", ".yaml", ".yml", ".ini"];
        if(directExtensions.includes(ext)) return await _streamToBuffer(readableStream); // direct content file
        // 2️. DOCX → Native text extraction (BEST)
        if (ext === ".docx" || ext === ".doc") return await _extractDocxText(await _streamToBuffer(readableStream));
        // 3️. PDF → OCR pipeline (PDF, scanned docs)
        if (ext === ".pdf") { 
            const result = await processPdfBase64WithoutPageMarkers(await _streamToBase64(readableStream));
            return Buffer.from(result.text);
        } else throw new Error(`Unsupported file extension for OCR: ${ext}`);
    } catch (err) {
        LOG.error(`getContent ERROR for file ${fileName}: ${err.message}`);
        return Buffer.from("");  // Return empty string on error
    }
}

/** Extract text buffer but return result as a readable stream */
async function getContentStream(readableStream, fileName) {
    try {
        const extractedTextBuffer = await getContent(readableStream, fileName);
        return Readable.from(extractedTextBuffer);  // Convert extracted text into a Node ReadableStream and return
    } catch (err) {
        LOG.error(`getContentStream ERROR for file ${fileName}: ${err.message}`);
        return Readable.from("");  // Return empty stream on error
    }
}

module.exports = { getContentStream, getContent };