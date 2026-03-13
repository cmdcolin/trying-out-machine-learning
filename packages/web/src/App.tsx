import { useState, useEffect, useRef } from 'react';
import './App.css';
import { runInference as runCpuInference } from '@mnist-jax/core';
import { WebGPUInference } from '@mnist-jax/webgpu';
import type { LayerWeights, TestExample } from '@mnist-jax/core';
import MNISTCanvas from './MNISTCanvas';

// Load assets
import weightsData from './assets/weights.json';
import testImagesData from './assets/test_images.json';

const weights = weightsData as LayerWeights[];
const testImages = testImagesData as TestExample[];

function App() {
  const [prediction, setPrediction] = useState<number | null>(null);
  const [probs, setProbs] = useState<number[]>([]);
  const [testResults, setTestResults] = useState<{label: number, pred: number}[]>([]);
  const [debugData, setDebugData] = useState<number[]>([]);
  const [useGpu, setUseGpu] = useState(false);
  const [gpuSupported, setGpuSupported] = useState(false);
  const gpuEngine = useRef<WebGPUInference | null>(null);
  const debugCanvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    // Check WebGPU support and init
    const initGpu = async () => {
      try {
        const engine = new WebGPUInference();
        await engine.init();
        gpuEngine.current = engine;
        setGpuSupported(true);
        setUseGpu(true); // Default to GPU if available
      } catch (e) {
        console.warn("WebGPU not available:", e);
        setGpuSupported(false);
      }
    };
    initGpu();

    // Run inference on all test images on load (CPU)
    const results = testImages.map(ex => ({
      label: ex.label,
      pred: runCpuInference(ex.image, weights).prediction
    }));
    setTestResults(results);
  }, []);

  useEffect(() => {
    if (debugCanvasRef.current && debugData.length === 784) {
      const ctx = debugCanvasRef.current.getContext('2d');
      if (ctx) {
        const imageData = ctx.createImageData(28, 28);
        for (let i = 0; i < 784; i++) {
          const val = debugData[i] * 255;
          imageData.data[i * 4] = val;
          imageData.data[i * 4 + 1] = val;
          imageData.data[i * 4 + 2] = val;
          imageData.data[i * 4 + 3] = 255;
        }
        ctx.putImageData(imageData, 0, 0);
      }
    }
  }, [debugData]);

  const handleDraw = async (data: number[]) => {
    setDebugData(data);
    
    if (useGpu && gpuEngine.current) {
      const { prediction: pred, probabilities } = await gpuEngine.current.runInference(data, weights);
      setPrediction(pred);
      setProbs(probabilities);
    } else {
      const { prediction: pred, probabilities } = runCpuInference(data, weights);
      setPrediction(pred);
      setProbs(probabilities);
    }
  };

  const handleClear = () => {
    setPrediction(null);
    setProbs([]);
    setDebugData([]);
  };

  return (
    <div className="App">
      <header className="app-header">
        <h1>MNIST JAX Demo</h1>
        <div className="mode-selector">
          <span className={!useGpu ? 'active' : ''}>CPU</span>
          <label className="switch">
            <input 
              type="checkbox" 
              checked={useGpu} 
              disabled={!gpuSupported}
              onChange={(e) => setUseGpu(e.target.checked)} 
            />
            <span className="slider round"></span>
          </label>
          <span className={useGpu ? 'active' : ''}>
            WebGPU {!gpuSupported && <span className="not-supported">(Not Supported)</span>}
          </span>
        </div>
      </header>
      
      <div className="container">
        <section className="demo-section">
          <h2>Draw a Digit</h2>
          <MNISTCanvas onDraw={handleDraw} onClear={handleClear} />
          
          <div className="debug-container">
            <div>
              <p>Normalized (28x28)</p>
              <canvas ref={debugCanvasRef} width={28} height={28} className="debug-canvas" />
            </div>
            {prediction !== null && (
              <div className="prediction-display">
                Prediction: <span className="winner">{prediction}</span>
              </div>
            )}
          </div>

          {probs.length > 0 && (
            <div className="prob-chart">
              {probs.map((p, i) => (
                <div key={i} className="prob-row">
                  <span className="digit-label">{i}</span>
                  <div className="prob-bar-container">
                    <div 
                      className={`prob-bar ${i === prediction ? 'winner-bar' : ''}`}
                      style={{ width: `${p * 100}%` }}
                    />
                  </div>
                  <span className="prob-value">{(p * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
          )}
        </section>

        <section className="test-section">
          <h2>Test Examples</h2>
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
    </div>
  );
}

export default App;
