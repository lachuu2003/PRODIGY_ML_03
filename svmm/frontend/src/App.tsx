import React, { useState } from 'react';
import axios from 'axios';

const App: React.FC = () => {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [prediction, setPrediction] = useState<string | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleImageChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedImage(file);
      setPreview(URL.createObjectURL(file));
      setPrediction(null);
      setError(null);
    }
  };

  const handlePredict = async () => {
    if (!selectedImage) {
      alert('Please upload an image first!');
      return;
    }

    const formData = new FormData();
    formData.append('image', selectedImage);

    // Log the FormData object to check if the image is correctly appended
    console.log('FormData:', formData);

    try {
      const response = await axios.post('https://prodigy-ml-03.onrender.com/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      console.log('Response:', response);  // Log the response to check what Flask returns
      if (response.data.prediction) {
        setPrediction(response.data.prediction);
        setError(null);
      } else {
        setError('No prediction returned');
      }
    } catch (error: any) {
      console.error('Error making prediction:', error);
      setError('Failed to make prediction. Please try again.');
    }
  };

  return (
    <div style={{ textAlign: 'center', padding: '20px' }}>
      <h1>Image Classifier</h1>
      <input type="file" accept="image/*" onChange={handleImageChange} />
      {preview && <img src={preview} alt="Preview" style={{ maxWidth: '200px', margin: '20px auto' }} />}
      <br />
      <button onClick={handlePredict} disabled={!selectedImage}>
        Predict
      </button>

      {prediction && (
        <div>
          <h3>Prediction:</h3>
          <p>{prediction}</p>
        </div>
      )}

      {error && (
        <div>
          <h3 style={{ color: 'red' }}>Error:</h3>
          <p>{error}</p>
        </div>
      )}
    </div>
  );
};

export default App;