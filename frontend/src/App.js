import React, { useState, useRef } from 'react';
import axios from 'axios'; 

// --- STYLES ---
const styles = {
  container: {
    fontFamily: '"Segoe UI", Roboto, Helvetica, Arial, sans-serif',
    backgroundColor: '#f8f9fa',
    minHeight: '100vh',
    color: '#333',
  },
  navbar: {
    backgroundColor: '#fff',
    padding: '15px 40px',
    boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    position: 'sticky',
    top: 0,
    zIndex: 1000,
  },
  logo: {
    fontSize: '24px',
    fontWeight: 'bold',
    color: '#007bff', // Medical Blue
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
  },
  hero: {
    backgroundImage: 'linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.7)), url("https://images.unsplash.com/photo-1579684385127-1ef15d508118?ixlib=rb-4.0.3&auto=format&fit=crop&w=1200&q=80")',
    backgroundSize: 'cover',
    backgroundPosition: 'center',
    height: '400px',
    display: 'flex',
    flexDirection: 'column',
    justifyContent: 'center',
    alignItems: 'center',
    color: 'white',
    textAlign: 'center',
    padding: '20px',
  },
  section: {
    maxWidth: '1000px',
    margin: '40px auto',
    padding: '0 20px',
  },
  card: {
    backgroundColor: 'white',
    borderRadius: '12px',
    padding: '30px',
    boxShadow: '0 4px 15px rgba(0,0,0,0.05)',
    marginBottom: '20px',
  },
  buttonPrimary: {
    backgroundColor: '#007bff',
    color: 'white',
    border: 'none',
    padding: '12px 30px',
    fontSize: '18px',
    borderRadius: '30px',
    cursor: 'pointer',
    transition: 'background 0.3s',
    fontWeight: '600',
    marginTop: '20px',
  },
  buttonSecondary: {
    backgroundColor: '#6c757d',
    color: 'white',
    border: 'none',
    padding: '10px 20px',
    borderRadius: '5px',
    cursor: 'pointer',
    marginTop: '10px',
  },
  grid: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: '40px',
    marginTop: '20px',
  },
  statusCard: (isHealthy) => ({
    backgroundColor: isHealthy ? '#d1e7dd' : '#f8d7da',
    color: isHealthy ? '#0f5132' : '#842029',
    padding: '20px',
    borderRadius: '10px',
    textAlign: 'center',
    marginBottom: '20px',
    border: `1px solid ${isHealthy ? '#badbcc' : '#f5c2c7'}`,
  }),
};

function App() {
  const [view, setView] = useState('home'); // 'home' or 'predict'
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const fileInputRef = useRef(null);

  // --- API LOGIC ---
  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      setResult(null);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      // Ensure this matches your backend URL/Port
      const response = await axios.post('http://127.0.0.1:8001/predict', formData);
      // Backend now returns { dr_result, other_condition, generalist_result, ai_advice }
      setResult(response.data);
    } catch (error) {
      console.error("Error:", error);
      alert("Error connecting to backend API. Is uvicorn running?");
    }
    setLoading(false);
  };

  const handleReset = () => {
    if (preview) {
      try { URL.revokeObjectURL(preview); } catch (e) { /* ignore */ }
    }
    setFile(null);
    setPreview(null);
    setResult(null);
    if (fileInputRef.current) fileInputRef.current.value = null;
  };

  // --- VIEWS ---

  const LandingPage = () => (
    <>
      <div style={styles.hero}>
        <h1 style={{ fontSize: '3.5rem', marginBottom: '10px' }}>Protect Your Vision</h1>
        <p style={{ fontSize: '1.2rem', maxWidth: '700px' }}>
          An advanced Cyber-Physical System using Fusion Deep Learning (VGG16 + ResNet50 + DenseNet121) 
          for early detection of Diabetic Retinopathy.
        </p>
        <button style={styles.buttonPrimary} onClick={() => setView('predict')}>
          Start Diagnosis Now
        </button>
      </div>

      <div style={styles.section}>
        <div style={styles.grid}>
          <div style={styles.card}>
            <h2 style={{color: '#007bff'}}>👁️ What is Diabetic Retinopathy?</h2>
            <p style={{lineHeight: '1.6'}}>
              Diabetic Retinopathy (DR) is a complication of diabetes that affects the eyes. 
              It's caused by damage to the blood vessels of the light-sensitive tissue at the 
              back of the eye (retina). Early detection is critical to prevent blindness.
            </p>
            <img 
              src="https://media.nature.com/lw767/magazine-assets/d41586-018-00004-w/d41586-018-00004-w_15324492.jpg" 
              alt="Retina" 
              style={{width: '100%', borderRadius: '10px', marginTop: '15px'}}
            />
            <p style={{fontSize: '0.8rem', color: '#666', marginTop: '5px'}}>Fig 1. Healthy Retina Structure</p>
          </div>

          <div style={styles.card}>
            <h2 style={{color: '#007bff'}}>🤖 How Our System Works</h2>
            <p style={{lineHeight: '1.6'}}>
              We utilize a <strong>Cyber-Physical Architecture</strong> where the image sensor data is processed 
              by a Fusion AI model in the cloud.
            </p>
            <ul style={{lineHeight: '1.8'}}>
              <li><strong>Step 1:</strong> Circular Cropping & Gaussian Blur preprocessing.</li>
              <li><strong>Step 2:</strong> Feature extraction via ResNet, VGG, and DenseNet.</li>
              <li><strong>Step 3:</strong> Attention-Guided feature fusion.</li>
              <li><strong>Step 4:</strong> Real-time diagnosis display.</li>
            </ul>
            <div style={{backgroundColor: '#e9ecef', padding: '15px', borderRadius: '8px', marginTop: '15px'}}>
              <strong>Accuracy:</strong> 90.25% <br/>
              <strong>Kappa Score:</strong> 0.812 (Strong Agreement)
            </div>
          </div>
        </div>
      </div>
    </>
  );

  const PredictionPage = () => (
    <div style={{
        backgroundImage: 'linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.7)), url("https://images.unsplash.com/photo-1579684385127-1ef15d508118?ixlib=rb-4.0.3&auto=format&fit=crop&w=1200&q=80")',
        backgroundSize: 'cover',
        backgroundPosition: 'center',
        backgroundAttachment: 'fixed',
        minHeight: 'calc(100vh - 74px)', 
        width: '100%',
        paddingTop: '1px' 
    }}>
      <div style={styles.section}>
        <button style={{...styles.buttonSecondary, marginBottom: '20px'}} onClick={() => setView('home')}>
          ← Back to Home
        </button>

        <div style={{display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '30px'}}>
          
          {/* LEFT: UPLOAD */}
          <div style={styles.card}>
            <h3 style={{marginTop: 0}}>1. Upload Patient Scan</h3>
            <div style={{
              border: '2px dashed #007bff', 
              borderRadius: '10px', 
              padding: '40px', 
              textAlign: 'center',
              backgroundColor: '#f8f9fa',
              cursor: 'pointer'
            }}>
              <input 
                ref={fileInputRef}
                type="file" 
                onChange={handleFileChange} 
                accept="image/*" 
                style={{display: 'none'}} 
                id="file-upload"
              />
              <label htmlFor="file-upload" style={{cursor: 'pointer', display: 'block'}}>
                {preview ? (
                  <img src={preview} alt="Preview" style={{maxWidth: '100%', maxHeight: '300px', borderRadius: '8px'}} />
                ) : (
                  <div>
                    <span style={{fontSize: '40px'}}>📂</span>
                    <p>Click to Upload Fundus Image</p>
                  </div>
                )}
              </label>
            </div>

            <button 
              style={{
                ...styles.buttonPrimary, 
                width: '100%', 
                opacity: (!file || loading) ? 0.6 : 1,
                cursor: (!file || loading) ? 'not-allowed' : 'pointer'
              }} 
              onClick={handleUpload}
              disabled={!file || loading}
            >
              {loading ? "Analyzing Retina..." : "Run AI Diagnosis"}
            </button>
          </div>

          {/* RIGHT: RESULTS */}
          <div style={styles.card}>
            <h3 style={{marginTop: 0}}>2. Analysis Results</h3>
            
            {!result && (
              <div style={{textAlign: 'center', padding: '40px', color: '#888'}}>
                <span style={{fontSize: '40px'}}>🩺</span>
                <p>Upload an image and run diagnosis to see results here.</p>
              </div>
            )}

            {result && (
              <div style={{animation: 'fadeIn 0.5s'}}>
                {(() => {
                  const dr = result.dr_result || {};
                  const other = result.other_condition || null;
                  const isHealthy = dr.diagnosis === 'No DR' || dr.diagnosis === 'No DR (Healthy)';
                  return (
                    <div style={{display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '12px'}}>
                      <div style={styles.statusCard(isHealthy)}>
                        <h2 style={{margin: 0}}>{dr.diagnosis}</h2>
                        <p style={{margin: '5px 0 0 0'}}>Confidence: {(dr.confidence * 100).toFixed(2)}%</p>
                      </div>

                      {/* Other condition badge */}
                      <div style={{minWidth: '160px', textAlign: 'right'}}>
                        {other ? (
                          <span style={{
                            display: 'inline-block',
                            padding: '8px 12px',
                            borderRadius: '999px',
                            backgroundColor: (other.label === 'Cataract' || other.label === 'Glaucoma') ? '#ffedd5' : '#e9f7ef',
                            color: (other.label === 'Cataract' || other.label === 'Glaucoma') ? '#92400e' : '#0f5132',
                            fontWeight: 600,
                            border: (other.label === 'Cataract' || other.label === 'Glaucoma') ? '1px solid #fbbf24' : '1px solid #b7e4c7'
                          }}>
                            {other.label} {other.confidence ? `(${(other.confidence*100).toFixed(1)}%)` : ''}
                          </span>
                        ) : (
                          <span style={{color: '#666'}}>Other condition: None</span>
                        )}
                      </div>
                    </div>
                  );
                })()}

                <h4>Detailed Probability Distribution:</h4>
                {Object.entries((result.dr_result && result.dr_result.probabilities) || {}).map(([stage, prob]) => (
                  <div key={stage} style={{marginBottom: '12px'}}>
                    <div style={{display: 'flex', justifyContent: 'space-between', fontSize: '14px', marginBottom: '4px'}}>
                      <span style={{fontWeight: stage === result.diagnosis ? 'bold' : 'normal'}}>{stage}</span>
                      <span>{(prob * 100).toFixed(1)}%</span>
                    </div>
                    <div style={{width: '100%', height: '8px', backgroundColor: '#e9ecef', borderRadius: '4px'}}>
                      <div style={{
                        width: `${prob * 100}%`,
                        height: '100%',
                        backgroundColor: stage === result.diagnosis ? '#007bff' : '#adb5bd',
                        borderRadius: '4px',
                        transition: 'width 1s ease-in-out'
                      }}></div>
                    </div>
                  </div>
                ))}

                <div style={{marginTop: '18px', display: 'flex', gap: '10px'}}>
                  <button
                    style={{...styles.buttonSecondary}}
                    onClick={() => {
                      handleReset();
                      if (fileInputRef.current) fileInputRef.current.click();
                    }}
                  >
                    Run Another Diagnosis
                  </button>

                  <button
                    style={{...styles.buttonSecondary}}
                    onClick={handleReset}
                  >
                    Clear
                  </button>
                </div>

                {/* Advice Box */}
                {result.ai_advice && (
                  <div style={{marginTop: '20px', padding: '16px', backgroundColor: '#fff3cd', borderRadius: '8px', border: '1px solid #ffeeba'}}>
                    <h4 style={{margin: '0 0 8px 0'}}>AI Advice</h4>
                    <p style={{margin: 0, color: '#7a4f01'}}>{result.ai_advice}</p>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <div style={styles.container}>
      <nav style={styles.navbar}>
        <div style={styles.logo}>
          <span>👁️</span> RetinaAI Pro
        </div>
        <div>
          <span style={{marginRight: '15px', color: 'green', fontWeight: 'bold'}}>● System Online</span>
        </div>
      </nav>

      {view === 'home' ? <LandingPage /> : <PredictionPage />}
    </div>
  );
}

export default App;