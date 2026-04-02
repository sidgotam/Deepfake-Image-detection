function uploadImage() {
    const input = document.getElementById('imageInput');
    const file = input.files[0];
    if (!file) {
      alert("Please select an image.");
      return;
    }
  
    // Preview
    const reader = new FileReader();
    reader.onload = () => {
      document.getElementById('preview').src = reader.result;
      document.getElementById('preview').style.display = 'block';
    };
    reader.readAsDataURL(file);
  
    const formData = new FormData();
    formData.append("image", file);
  
    fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      body: formData
    })
    .then(res => res.json())
    .then(data => {
      document.getElementById('result').innerHTML = `
        <h2>Prediction: ${data.prediction}</h2>
        <p>Confidence: ${data.confidence}%</p>
      `;
    })
    .catch(err => {
      console.error(err);
      alert("Failed to get a prediction.");
    });
  }
  