
let fs=require('fs')
let fsPath=require('fs-path')
let dataBase=fsPath.findSync('./data').files

let {encode,decode}=require('gpt-3-encoder')
const path = require('path');
const fastText = require('fasttext');
 

 



const { pipeline } = require('node:stream/promises');

async function  processChunk(line,{signal}){

    fs.appendFileSync('./models/base',encode(line).join(' ')+'\n')
}

async function digest(data) {
  await pipeline(
    fs.createReadStream(data),
    async function* (source, { signal }) {
      source.setEncoding('utf8');  
      for await (const line of source) {
        yield await  processChunk(line, { signal });
      }
    },
    
  );
    return true
}

async function digestData(params) {
    for(let k=0;k<dataBase.length;k++){
        await digest(dataBase[k])
        console.log(k)
    }
    let data = path.resolve(path.join(__dirname, './models/base'));
let model = path.resolve(path.join(__dirname, './models/vec'));
    let classifier = new fastText.Classifier();
let options = {
    input: data,
    output: model,
    ws:5,
    neg:1,
    epoch:50,
    dim:8,
    bucket: 2000000
}
 
classifier.train('cbow', options)
}

digestData()