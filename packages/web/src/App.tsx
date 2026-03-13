import { useState, useEffect } from 'react';
import './App.css';
import { runInference } from './inference';
import type { LayerWeights, TestExample } from './inference';
import MNISTCanvas from './MNISTCanvas';

// Load assets
import weightsData from './assets/weights.json';
import testImagesData from './assets/test_images.json';

const weights = weightsData as LayerWeights[];
const testImages = testImagesData as TestExample[];

function App() {
  const [prediction, setPrediction] = useState<number | null>(null);
  const [testResults, setTestResults] = useState<{label: number, pred: number}[]>([]);

  useEffect(() => {
    // Run inference on all test images on load
    const results = testImages.map(ex => ({
      label: ex.label,
      pred: runInference(ex.image, weights)
    }));
    setTestResults(results);
  }, []);

  const handleDraw = (data: number[]) => {
    const pred = runInference(data, weights);
    setPrediction(pred);
  };

  const handleClear = () => {
    setPrediction(null);
  };

  return (
    <div className="App">
      <h1>MNIST JAX Demo</h1>
      
      <div className="container">
        <section className="demo-section">
          <h2>Draw a Digit</h2>
          <p>Draw a digit (0-9) in the box below</p>
          <MNISTCanvas onDraw={handleDraw} onClear={handleClear} />
          {prediction !== null && (
            <div className="prediction">
              Prediction: <span>{prediction}</span>
            </div>
          )}
        </section>

        <section className="test-section">
          <h2>Test Examples</h2>
          <p>Pre-loaded examples from MNIST test set</p>
          <div className="test-grid">
            {testResults.map((res, i) => (
              <div key={i} className="test-item">
                <div className="test-label">True: {res.label}</div>
                <div className={`test-pred ${res.label === res.pred ? 'correct' : 'wrong'}`}>
                  Pred: {res.pred}
                </div>
              </div>
            ))}
          </div>
        </section>
      </div>

      <footer style={{ marginTop: '40px', color: '#888' }}>
        <p>Trained with JAX. Inference in native TypeScript.</p>
      </footer>
    </div>
  );
}

export default App;
