import * as fs from 'fs';
import * as path from 'path';

// --- Types ---
type Matrix = number[][];
type Vector = number[];

interface LayerWeights {
    weights: Matrix; // [out_features, in_features]
    biases: Vector;  // [out_features]
}

interface TestExample {
    image: Vector;
    label: number;
}

// --- Linear Algebra Helpers ---
// Note: This is a naive implementation for demonstration purposes.

function dot(v1: Vector, v2: Vector): number {
    let sum = 0;
    for (let i = 0; i < v1.length; i++) {
        sum += v1[i] * v2[i];
    }
    return sum;
}

function matMulVec(m: Matrix, v: Vector): Vector {
    // m is [rows, cols], v is [cols]
    // result is [rows]
    const rows = m.length;
    const result = new Array(rows);
    for (let i = 0; i < rows; i++) {
        result[i] = dot(m[i], v);
    }
    return result;
}

function addVec(v1: Vector, v2: Vector): Vector {
    if (v1.length !== v2.length) throw new Error("Vector length mismatch");
    const result = new Array(v1.length);
    for (let i = 0; i < v1.length; i++) {
        result[i] = v1[i] + v2[i];
    }
    return result;
}

function relu(v: Vector): Vector {
    return v.map(val => Math.max(0, val));
}

function argmax(v: Vector): number {
    let maxVal = -Infinity;
    let maxIdx = -1;
    for (let i = 0; i < v.length; i++) {
        if (v[i] > maxVal) {
            maxVal = v[i];
            maxIdx = i;
        }
    }
    return maxIdx;
}

// --- Inference ---

function predict(image: Vector, layers: LayerWeights[]): number {
    let activations = image;

    for (let i = 0; i < layers.length; i++) {
        const { weights, biases } = layers[i];
        
        // Linear layer: Wx + b
        // weights are [out, in], image is [in]
        let output = matMulVec(weights, activations);
        output = addVec(output, biases);

        // Activation (ReLU for hidden layers, LogSoftmax/Identity for output)
        // The last layer in our JAX model was just a linear layer followed by LogSoftmax in loss
        // But for prediction (argmax), we don't strictly need softmax if we just want the max.
        // However, the hidden layers need ReLU.
        
        if (i < layers.length - 1) {
            activations = relu(output);
        } else {
            // Output layer
            activations = output;
        }
    }

    return argmax(activations);
}

// --- Main ---

function main() {
    const weightsPath = path.join(__dirname, 'weights.json');
    const testDataPath = path.join(__dirname, 'test_images.json');

    if (!fs.existsSync(weightsPath) || !fs.existsSync(testDataPath)) {
        console.error("Error: weights.json or test_images.json not found. Run train.py first.");
        process.exit(1);
    }

    console.log("Loading weights and test data...");
    const layers: LayerWeights[] = JSON.parse(fs.readFileSync(weightsPath, 'utf8'));
    const testExamples: TestExample[] = JSON.parse(fs.readFileSync(testDataPath, 'utf8'));

    console.log(`Loaded ${layers.length} layers.`);
    console.log(`Running inference on ${testExamples.length} examples...\n`);

    let correct = 0;
    for (let i = 0; i < testExamples.length; i++) {
        const { image, label } = testExamples[i];
        const prediction = predict(image, layers);
        
        const isCorrect = prediction === label;
        if (isCorrect) correct++;

        console.log(`Example ${i + 1}:`);
        console.log(`  True Label: ${label}`);
        console.log(`  Prediction: ${prediction}`);
        console.log(`  Result:     ${isCorrect ? "PASS" : "FAIL"}`);
        console.log("-------------------------");
    }

    console.log(`Accuracy on demo set: ${correct}/${testExamples.length} (${(correct / testExamples.length) * 100}%)`);
}

main();
