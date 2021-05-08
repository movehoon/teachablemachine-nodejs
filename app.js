const canvas = require('canvas');
require('@tensorflow/tfjs'); // Load before @teachablemachine/image
const tmImage = require('@teachablemachine/image')
const express = require('express');
const formData = require('express-form-data');
const os = require('os');
const fs = require('fs')

const app = express();

const options = {
    uploadDir: os.tmpdir(),
    autoClean: true
}

app.use(formData.parse(options))
app.use(formData.format())
app.use(formData.stream())
app.use(formData.union())
app.use(express.urlencoded({
    extended: true
}))
app.use(express.json())
app.use(require('body-parser').raw({ type: 'image/png', limit: '3MB' }));


const JSDOM = require('jsdom').JSDOM;
const { loadGraphModel } = require('@tensorflow/tfjs-converter');
global.window = new JSDOM(`<body><script>document.body.appendChild(document.createElement("hr"));</script></body>`).window;
global.document = window.document;
global.fetch = require('node-fetch');

app.listen(3000, () => {
    console.log("Server running on port 3000");
});

let models = {}

async function loadModel(URL) {
    if (models[URL] == undefined) {
        console.log('load model from ', URL);
        const modelURL = URL + 'model.json';
        const metadataURL = URL + 'metadata.json';
        models[URL] = await tmImage.load(modelURL, metadataURL);
    }
}

app.post('/load', (req, res, next) => {
    let url = req.body['url']
    console.log('load ', url)
    loadModel(url)
    res.send('load' + url)
})

app.post('/predict', (req, res, next) => {
    // console.log('req.body ', req.body)
    // console.log('req.body["url"] ', req.body['url'])
    // console.log('req.body["image"] ', req.body['image'])
    let url = req.body['url']
    let image = req.body['image']
    // console.log('url', url)
    // console.log('path', image['path'])
    fs.readFile(image['path'], function(err, data) {
        if (err) throw err;
        // console.log(data);

        if (models[url] != undefined) {
            try {
                getPrediction(models[url], _arrayBufferToBase64(data), (output) => {
                    res.send(output);
                });
            }
            catch (error) {
                res.send('Error on processing')
            }
        }
        else {
            loadModel(url)
            res.send('Model not loaded')
        }
    });
});

// loadModel('https://teachablemachine.withgoogle.com/models/ymwzDuQDL/');
// loadModel('https://teachablemachine.withgoogle.com/models/BkzXnW7fb/');

async function getPrediction(model, data, fu) {
    const can = canvas.createCanvas(64, 64);
    const ctx = can.getContext('2d');

    const img = new canvas.Image();
    img.onload = async () => {
        ctx.drawImage(img, 0, 0, 64, 64);
        const prediction = await model.predict(can);
        console.log(prediction);
        fu(prediction);
    }
    img.onerror = err => { throw err; }
    img.src = "data:image/png;base64," + data;
}

function _arrayBufferToBase64( buffer ) {
    var binary = '';
    var bytes = new Uint8Array( buffer );
    var len = bytes.byteLength;
    for (var i = 0; i < len; i++) {
        binary += String.fromCharCode( bytes[ i ] );
    }
    return window.btoa( binary );
}