const canvas = require('canvas');
require('@tensorflow/tfjs'); // Load before @teachablemachine/image
const tmImage = require('@teachablemachine/image')
const tmPose = require('@teachablemachine/pose')
const express = require('express');
const formData = require('express-form-data');
const os = require('os');
const fs = require('fs')
const path = require('path')

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
let models_pose = {}
let model_mask = undefined

async function loadModelMask() {
    // model_mask = await tmImage.load('./mask_model/model.json', './mask_model/metadata.json')
    let modelURL = path.join(__dirname, '/mask_model/model.json')
    let weightURL = path.join(__dirname, '/mask_model/weights.bin')
    let metadataURL = path.join(__dirname, '/mask_model/metadata.json')
    console.log('modelURL: ', modelURL)
    model_mask = await tmImage.loadFromFiles(
        modelURL,
        weightURL,
        metadataURL
    )
    // model_mask = await tmImage.load(
    //     modelURL,
    //     metadataURL
    // )
}
// model_mask = loadModelMask()

async function loadModel(URL) {

    if (models[URL] != undefined) {
        delete models[URL]
    }
    console.log('load model from ', URL);
    const modelURL = URL + 'model.json';
    const metadataURL = URL + 'metadata.json';
    models[URL] = await tmImage.load(modelURL, metadataURL);
    console.log('model loaded ', URL);
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
                    var prb = 0
                    var idx = 0
                    var cls = 'NOT_FOUND'
                    for (var i=0; i<output.length; i++) {
                        console.log('output: ', i, ' => ', output[i])
                        if (output[i]['probability'] > prb) {
                            prb = output[i]['probability']
                            cls = output[i]['className']
                            idx = i
                        }
                    }
                    console.log('prb: ', prb)
                    console.log('class: ', cls)
                    console.log('index: ', idx)
                    res.send(cls)
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

app.post('/predict_mask', (req, res, next) => {
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

        if (model_mask != undefined) {
            try {
                getPrediction(model_mask, _arrayBufferToBase64(data), (output) => {
                    var prb = 0
                    var idx = 0
                    var cls = 'NOT_FOUND'
                    for (var i=0; i<output.length; i++) {
                        console.log('output: ', i, ' => ', output[i])
                        if (output[i]['probability'] > prb) {
                            prb = output[i]['probability']
                            cls = output[i]['className']
                            idx = i
                        }
                    }
                    console.log('prb: ', prb)
                    console.log('class: ', cls)
                    console.log('index: ', idx)
                    res.send(cls)
                });
            }
            catch (error) {
                res.send('Error on processing')
            }
        }
        else {
            loadModelMask()
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


// ----------------
// ----- Pose -----
// ----------------

async function loadModelPose(URL) {
    console.log('loadModelPose')
    if (models_pose[URL] != undefined) {
        delete models_pose[URL]
    }
    console.log('load model pose from ', URL);
    const modelURL = URL + 'model.json';
    const metadataURL = URL + 'metadata.json';
    models_pose[URL] = await tmPose.load(modelURL, metadataURL);
    console.log('model loaded ', URL);
}

app.post('/load_pose', (req, res, next) => {
    let url = req.body['url']
    console.log('load pose ', url)
    loadModelPose(url)
    res.send('load' + url)
})


app.post('/predict_pose', (req, res, next) => {
    // console.log('/predict_pose rereq.body ', req.body)
    // console.log('req.body["url"] ', req.body['url'])
    // console.log('req.body["image"] ', req.body['image'])
    let url = req.body['url']
    let image = req.body['image']
    // console.log('url', url)
    // console.log('path', image['path'])
    fs.readFile(image['path'], function(err, data) {
        if (err) throw err;
        // console.log(data);

        if (models_pose[url] != undefined) {
            try {
                getPredictionPose(models_pose[url], _arrayBufferToBase64(data), (output) => {
                    var prb = 0
                    var idx = 0
                    var cls = 'NOT_FOUND'
                    for (var i=0; i<output.length; i++) {
                        console.log('output: ', i, ' => ', output[i])
                        if (output[i]['probability'] > prb) {
                            prb = output[i]['probability']
                            cls = output[i]['className']
                            idx = i
                        }
                    }
                    console.log('prb: ', prb)
                    console.log('class: ', cls)
                    console.log('index: ', idx)
                    res.send(cls)
                });
            }
            catch (error) {
                res.send('Error on processing')
            }
        }
        else {
            loadModelPose(url)
            res.send('Model pose not loaded')
        }
    });
});

async function getPredictionPose(model, data, fu) {
    const can = canvas.createCanvas(64, 64);
    const ctx = can.getContext('2d');

    const img = new canvas.Image();
    img.onload = async () => {
        ctx.drawImage(img, 0, 0, 64, 64);
        const { pose, posenetOutput } = await model.estimatePose(can)
        const prediction = await model.predict(posenetOutput);
        console.log(prediction);
        fu(prediction);
    }
    img.onerror = err => { throw err; }
    img.src = "data:image/png;base64," + data;
}
