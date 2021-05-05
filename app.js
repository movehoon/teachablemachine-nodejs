const express = require('express');
// const formData = require('express-form-data');
// const bodyParser = require('body-parser')
// const os = require('os');
// var multer = require('multer');
const tf = require('@tensorflow/tfjs');
const tmImage = require('@teachablemachine/image');
// global.fetch = require('node-fetch')
const canvas = require('canvas');
// const { fstat } = require('fs');
const app = express();
const PORT = 4000;

const JSDOM = require('jsdom').JSDOM;
global.window = new JSDOM(`<body><script>document.body.appendChild(document.createElement("hr"));</script></body>`).window;
global.document = window.document;
global.fetch = require('node-fetch');

const MODEL_URL = 'https://teachablemachine.withgoogle.com/models/ymwzDuQDL/'
const MODEL_URL2 = 'https://teachablemachine.withgoogle.com/models/BkzXnW7fb/'

const options = {
    uploadDir: os.tmpdir(),
    autoClean: true
}

var dicModel = {}
let model, webcam, labelContainer, maxPredictions;

function hasModel(key) {
    if (dicModel[key] != undefined)
        return true
    console.log(key, ' is not exist')
    return false
}

async function initModel(url) {
    const modelURL = url + 'model.json';
    const metadataURL = url + 'metadata.json';

    // dicModel[url] = await tmImage.load(modelURL, metadataURL);
    model = await tmImage.load(modelURL, metadataURL);
    maxPredictions = model.getTotalClasses();
}

async function setupModel(key) {
    if (!hasModel(key)) {
        // init model
        initModel(key)
    }
}

// async function predict(image) {
//     const buffer = canvas.toBuffer('image/png')
//     fstat.writeFileSync()
//     const canvas = createCanvas(200, 200)

//     const prediction = await model.predict(createImageBitmap(image))
//     for (let i=0; i<maxPredictions; i++) {
//         console.log(prediction[i].name, ': ', prediction[i].probability.toFixed(2))
//     }
// }



async function getPrediction(data, fu) {
    const can = canvas.createCanvas(64, 64);
    const ctx = can.getContext('2d');

    const img = new canvas.Image();
    console.log('1')
    img.onload = async () => {
        ctx.drawImage(img, 0, 0, 64, 64);
        console.log('2')

        const prediction = await model.predict(can);
        console.log(prediction);
        fu(prediction);
        console.log('3')
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

// setupModel(MODEL_URL);

const handleListening = () => {
    console.log(`Listening on: http://localhost:${PORT}`)
}

const handleHome = (req, res) => {
    res.send('Hello from home')
}

const handleProfile = (req, res) => {
    res.send('You are on my profile')
}

const handleSetup = (req, res) => {
    console.log('handleSetup ')
    console.log('req.body.url ', req.body.url)

    if (req.body.url.length>0) {
        setupModel(req.body.url)
    }

    res.send('Setup')
}

const handleClassify = (req, res) => {
    // console.log('handleClassify ', req)
    // console.log('req.get ', req.get)
    // console.log('req.form ', req.form)
    // console.log('req.files ', req.files)
    console.log('req.body ', req.body)
    // console.log('req.body.image ', req.body.image)
    // console.log('req.body["image"] ', req.files['image'])
    getPrediction(model, _arrayBufferToBase64(req.body), (output) => {
        res.send(output);
    });
    // fs.writeFile('./image.png', req.body['image'], function(err) {
    //     if (err) throw err;
    // })
    // predict(req.files['image'])
    // const { url } = req.files;

    // return model.classify({
    //     imageUrl: url,
    // }).then((predictions) => {
    //     console.log(predications);
    //     return res.json(predictions);
    // }).catch((e) => {
    //     console.error(e);
    //     res.status(500).send('Something went wrong')
    // });
//   res.send('Classify')
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

// app.use(multer().array())

app.get('/profile', handleProfile)
app.get('/', handleHome)
app.post('/setup', handleSetup)
app.post('/classify', handleClassify)
app.listen(4000,handleListening);

setupModel(MODEL_URL)
