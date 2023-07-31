let fs = require('fs')
let tf = require('@tensorflow/tfjs-node')//gpu let tf = require('@tensorflow/tfjs-node-gpu')

let euclidean = require('talisman/metrics/euclidean')
let fsPath = require('fs-path')
let _ = require('lodash')
let gpt3 = require('gpt-3-encoder')
let cfg={
    sequenceSize:512,
    predictSteps:8  ,
    dimension:512,
    batchSize:32,
    arrayDimension:8
    
}


let v = fs.readFileSync('./models/vec.vec').toString()
v = v.split('\n')
v.shift()
let w2v = {}
v = v.map(s => {
    let y = s.split(' ')

    let key = y.shift()
    let vec = y.filter(s => s.length).map(s => s / 1)

    w2v[key] = vec

})
let vw2v = _.toPairs(w2v)

function convert(n) {
    let k = w2v[n] || new Array(8).fill(0)


    return k
}



function reverser(arr) {
    let max = 0
    let best = 100000
    for (let k = 0; k < vw2v.length; k++) {

        let v = vw2v[k][1] && vw2v[k][1][0] ? vw2v[k][1] : Array(8).fill(0)

        let b = euclidean(v, arr)


        if (b < best) {
            best = b
            max = vw2v[k][0]
        }
    }
    return max
}


class TransformerLayer extends tf.layers.Layer {

    static className = "TransformerLayer";
    inputDenseWeight = null;
    inputDenseBias = null;
    queryDenseWeight = null;
    queryDenseBias = null;
    keyDenseWeight = null;
    keyDenseBias = null;
    valueDenseWeight = null;
    valueDenseBias = null;
    denseWeight = null;
    denseBias = null;
    ffDense1Weight = null;
    ffDense1Bias = null;
    ffDense2Weight = null;
    ffDense2Bias = null;
    DEFAULT_KERNEL_INITIALIZER = "glorotNormal";

    constructor(args = {
        numHeads: 8,
        padSize: opt.lenth,
        depth: opt.depth
    }) {
        super(args);
        this.numHeads = args.numHeads;
        this.depth = args.depth;
        this.pool = args.pool;
        this.padSize = args.padSize;

        if (this.depth % this.numHeads != 0) {
            throw new Error(`erro : depth(${this.depth}) != numHead(${this.numHeads}) != 0 `);
        }

        this.inputDenseWeight = null;
        this.inputDenseBias = null;
        this.queryDenseWeight = null;

        this.queryDenseBias = null;
        this.keyDenseWeight = null;
        this.keyDenseBias = null;
        this.valueDenseWeight = null;
        this.valueDenseBias = null;
        this.denseWeight = null;
        this.denseBias = null;
        this.ffDense1Weight = null;
        this.ffDense1Bias = null;
        this.ffDense2Weight = null;
        this.ffDense2Bias = null;
        this.weightsInitializer = tf.initializers.glorotUniform(this.inputShape)

    }

    build(inputShape) {

        const inputFeatSize = inputShape[inputShape.length - 1];
        this.randomIdAtt = this.addWeight("randomIdAtt", [1], "float32", tf.initializers.zeros([1]), undefined, true);
        this.randomIdFF = this.addWeight("randomIdFF", [1], "float32", tf.initializers.zeros([1]), undefined, true);
        this.inputDenseWeight = this.addWeight("inputDenseWeight", [inputFeatSize, this.depth], "float32", this.weightsInitializer, undefined, true);
        this.inputDenseBias = this.addWeight("inputDenseBias", [this.depth], "float32", this.weightsInitializer, undefined, true);
        this.queryDenseWeight = this.addWeight("queryDenseWeight", [this.depth, this.depth], "float32", this.weightsInitializer, undefined, true);
        this.queryDenseBias = this.addWeight("queryDenseBias", [this.depth], "float32", this.weightsInitializer, undefined, true);
        this.keyDenseWeight = this.addWeight("keyDenseWeight", [this.depth, this.depth], "float32", this.weightsInitializer, undefined, true);
        this.keyDenseBias = this.addWeight("keyDenseBias", [this.depth], "float32", this.weightsInitializer, undefined, true);
        this.valueDenseWeight = this.addWeight("valueDenseWeight", [this.depth, this.depth], "float32", this.weightsInitializer, undefined, true);
        this.valueDenseBias = this.addWeight("valueDenseBias", [this.depth], "float32", this.weightsInitializer, undefined, true);
        this.denseWeight = this.addWeight("denseWeight", [this.depth, this.depth], "float32", this.weightsInitializer, undefined, true);
        this.denseBias = this.addWeight("denseBias", [this.depth], "float32", this.weightsInitializer, undefined, true);
        this.ffDense1Weight = this.addWeight("ffDense1Weight", [this.depth, this.depth], "float32", this.weightsInitializer, undefined, true);
        this.ffDense1Bias = this.addWeight("ffDense1Bias", [this.depth], "float32", this.weightsInitializer, undefined, true);
        this.ffDense2Weight = this.addWeight("ffDense2Weight", [this.depth, this.depth], "float32", this.weightsInitializer, undefined, true);
        this.ffDense2Bias = this.addWeight("ffDense2Bias", [this.depth], "float32", this.weightsInitializer, undefined, true);

        this.built = true;
    }

    computeOutputShape(inputShape) {
        if (this.pool) {
            return [inputShape[0], this.depth];
        } else {
            return [inputShape[0], inputShape[1], this.depth];
        }
    }

    call(inputs, kwargs) {

        return tf.tidy(() => {
            this.invokeCallHook(inputs, kwargs);
            let K = tf
            const batchSize = inputs[0].shape[0]


            const flatInput = inputs[0].reshape([this.padSize * batchSize, -1]);

            const flatScaledInput = K.dot(flatInput, this.inputDenseWeight.read()).add(this.inputDenseBias.read());

            const scaledInput = flatScaledInput.reshape([batchSize, this.padSize, -1]);


            const flatQuery = K.dot(flatScaledInput, this.queryDenseWeight.read()).add(this.queryDenseBias.read());

            const flatKey = K.dot(flatScaledInput, this.keyDenseWeight.read()).add(this.keyDenseBias.read());

            const flatValue = K.dot(flatScaledInput, this.valueDenseWeight.read()).add(this.valueDenseBias.read());

            const query = flatQuery.reshape([batchSize, this.padSize, -1]);

            const key = flatKey.reshape([batchSize, this.padSize, -1]);

            const value = flatValue.reshape([batchSize, this.padSize, -1]);

            const queryT = tf.transpose(query.reshape([batchSize, -1, this.numHeads, this.depth / this.numHeads]), [0, 2, 1, 3]);

            const keyT = tf.transpose(key.reshape([batchSize, -1, this.numHeads, this.depth / this.numHeads]), [0, 2, 1, 3]);

            const valueT = tf.transpose(value.reshape([batchSize, -1, this.numHeads, this.depth / this.numHeads]), [0, 2, 1, 3]);



            const matmul_qk = tf.matMul(queryT, keyT, false, true);

            let logits = matmul_qk.div(tf.sqrt(tf.cast(this.depth, "float32")));

            const toBroadcastMask = tf.ones([batchSize, this.padSize]).expandDims(1).expandDims(1)

            logits = logits.add(tf.scalar(1.0).sub(toBroadcastMask).mul(-1e9));

            let attentionWeights = logits.sigmoid().mul(logits.tanh())


            const scaledAttention = tf.matMul(attentionWeights, valueT, true, false);

            const scaledAttentionT = tf.transpose(scaledAttention, [0, 2, 1, 3]);
            const concatAttention = scaledAttentionT.reshape([scaledAttentionT.shape[0], -1, this.depth]);
            const flattenConcatAttention = concatAttention.reshape([batchSize * this.padSize, -1]);
            const flattenAttention = K.dot(flattenConcatAttention, this.denseWeight.read()).add(this.denseBias.read());
            const attention = flattenAttention.reshape([batchSize, this.padSize, -1]);

            const normalizedLatent = (scaledInput).add(this.randomIdAtt.read().squeeze(-1).mul(attention))


            const flattenNormalizedLatent = normalizedLatent.reshape([batchSize * this.padSize, -1]);
            const flatFf1 = K.dot(flattenNormalizedLatent, this.ffDense1Weight.read()).add(this.ffDense1Bias.read());
            const flatRff1 = tf.leakyRelu(flatFf1);
            const flatFf2 = K.dot(flatRff1, this.ffDense2Weight.read()).add(this.ffDense2Bias.read());
            const flatDff2 = tf.dropout(flatFf2, 0.1);
            const dff2 = flatDff2.reshape([batchSize, this.padSize, -1]);

            const output = normalizedLatent.add((this.randomIdFF.read().squeeze(-1)).mul(dff2));

            if (this.pool) {
                return output.mean(1);
            } else {
                return output
            }
        });
    }

    getConfig() {
        const config = super.getConfig();
        Object.assign(config, {
            pool: this.pool,
            padSize: this.padSize,
            numHeads: this.numHeads,
            depth: this.depth
        });
        return config;
    }

}

function addler(data, mod) {

    let MOD_ADLER = mod

    data = data.map(s => s)

    let len = data.length
    let a = 1,
        b = 0;
    let index;


    for (index = 0; index < len; index++) {
        a = (a + data[index]) % MOD_ADLER;
        b = (b + a) % MOD_ADLER;


    }
    return b
}
tf.serialization.registerClass(TransformerLayer);



let words = {}
let maxf = 0
let books = _.shuffle(fsPath.findSync('./data').files)

function list(v) {
    v = v.map((s, i) => [s, i])
    return [v.map(s => s[0]), v.map(s => (s[1]))]
}

function reparse(v) {
    v = v.map((s, i) => [(s).length ? reverser(s):s])

    return gpt3.decode(v.map(s => s[0]))
}

async function run(params) {

    let input = tf.layers.input({
        shape: [cfg.sequenceSize, 8]
    })
    let input2 = tf.layers.input({
        shape: [cfg.sequenceSize]
    })
   



    let x = tf.layers.permute({
        dims: [2, 1]
    }).apply(input)




    let skip = x

    x = tf.layers.conv1d({
        filters: cfg.dimension,
        kernelSize: 1,
        strides: 1,
        padding: "same",
        activation: "mish"
    }).apply(x)
    x = new TransformerLayer({
        depth: cfg.dimension,
        numHeads: 4,
        padSize: cfg.arrayDimension
    }).apply(x)
    x = new TransformerLayer({
        depth: cfg.sequenceSize,
        numHeads: 4,
        padSize: cfg.arrayDimension
    }).apply(x)


    let x3a = tf.layers.permute({
        dims: [2, 1]
    }).apply(x)



    x1 = tf.layers.dense({
        units: cfg.arrayDimension,
        activation: "linear"
    }).apply(x3a)

    let model = tf.model({
        inputs: [input],
        outputs: [x1]
    })
    model.compile({
        loss: [tf.losses.huberLoss],
        metrics: ['acc'],
        optimizer: tf.train.adam(.0005)
    })
    model.summary()

    let n = 0
    let setx = []
    let sety = []
    for (let bookIndex = 0; bookIndex < books.length; bookIndex++) {

        let book = _.chunk(fs.readFileSync(books[bookIndex]).toString().split(' '), cfg.sequenceSize).map(s => gpt3.encode(s.join(' ')))

        book = _.flatten(book)
      
        let pool = book.splice(0, cfg.sequenceSize*2)






        function makeIterator() {
            const numElements = book.length
            let index = 0;
            let n = 0
            const iterator = {
                next: x => tf.tidy(() => {
                    let result;
         
                    setx = []
                    n++
       
                    for (let k = 0; k < cfg.batchSize; k++) {

                        pool = book.slice(n + k, k + n + cfg.sequenceSize + cfg.predictSteps)


                        let xs = list(_.take(pool, cfg.sequenceSize))
                        let ys = list(_.takeRight(pool, cfg.sequenceSize))
                        setx.push([xs[0].map(s => convert(s)),  ys[0].map(s => convert(s))])

                     


                    }
                    if (index < numElements) {
                        let tx1 = tf.tensor(setx.map(s => s[0]))

                        let ty1 = tf.tensor(setx.map(s => s[1]))
                        result = {
                            value: {
                                xs: tx1,
                                ys: ty1
                            },
                            done: false
                        };

                        return result;
                    }
                    return {
                        value: index,
                        done: true
                    };
                })
            }

            return iterator;
        }



        const ds = tf.data.generator(makeIterator);
        await model.fitDataset(ds, {
            batchesPerEpoch: 128,
            epochs: Math.round(book.length / 128),
            callbacks: {
                async onEpochEnd() {
                        await model.save('file://./models/llm')
                    tf.tidy(() => {
                        let r = _.random(0, setx.length - 1)

                        let s = [setx[r]]
                        let tx11 = tf.tensor(s.map(s => s[0]))
                      
                        let res = model.predict(tx11)
                        let a = res.arraySync()[0]

        
                        console.clear()
                        console.log('---------------------------------INPUT-----------------------------------------')
                        console.log(reparse(setx[r][0]))
                        console.log('---------------------------------REAL-----------------------------------------')
                        console.log(reparse(setx[r][1]))
                        console.log('--------------------------------PREDICT----------------------------------------')
                        console.log(gpt3.decode(a.map(s => (reverser(s)))))
                        res.dispose()

                        tx11.dispose()
                       
                    })
                }
            }
        })

        setx = []
    }



}




run()