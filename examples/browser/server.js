const http = require('http');
const fs = require('fs');
const path = require('path');

const server = http.createServer((req, res) => {
  // Set Cross-Origin-Opener-Policy and Cross-Origin-Embedder-Policy headers
  res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
  res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');

  // Extract the file path from the request URL
  let filePath = '.' + req.url;
    if (path.normalize(decodeURI(req.url)) !== decodeURI(req.url)) {
        res.statusCode = 403;
        res.end();
        return;
    } 

  // If the URL is a directory, serve the 'index.html' file within it
  if (filePath === './') {
    filePath = './index.html';
  }

  // Resolve the absolute file path
  filePath = path.resolve(filePath);

  // Check if the file exists
  fs.access(filePath, fs.constants.F_OK, (err) => {
    if (err) {
      // File not found
      res.writeHead(404);
      res.end('File not found');
      return;
    }

    // Read the file and send it in the response
    fs.readFile(filePath, (err, data) => {
      if (err) {
        res.writeHead(500);
        res.end('Error reading file');
        return;
      }

      // Set the appropriate content type based on the file extension
      const extname = path.extname(filePath);
      const contentType = getContentType(extname);
      res.setHeader('Content-Type', contentType);

      // Send the file data in the response
      res.writeHead(200);
      res.end(data);
    });
  });
});

// Start the server on port 3000
server.listen(3000, () => {
  console.log('Server running on http://localhost:3000/');
});

// Helper function to get the content type based on the file extension
function getContentType(extname) {
  switch (extname) {
    case '.html':
      return 'text/html';
    case '.css':
      return 'text/css';
    case '.js':
      return 'text/javascript';
    case '.json':
      return 'application/json';
    case '.png':
      return 'image/png';
    case '.jpg':
      return 'image/jpeg';
    case '.wasm':
      return 'application/wasm'; // Set 'application/wasm' for .wasm files
    default:
      return 'application/octet-stream';
  }
}
