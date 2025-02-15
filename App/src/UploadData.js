import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Upload, FileText, ArrowRight, CheckCircle } from 'lucide-react';

const UploadPage = () => {
  const [files, setFiles] = useState([]);
  const [processing, setProcessing] = useState(false);
  const [success, setSuccess] = useState(false);
  const navigate = useNavigate();

  const handleFileChange = (e) => {
    setFiles(Array.from(e.target.files));
  };

  const handleUpload = async () => {
    if (files.length === 0) return;
    setProcessing(true);
    const formData = new FormData();
    files.forEach(file => {
      formData.append('files', file);
    });

    try {
      const response = await fetch('http://localhost:5000/upload', {
        method: 'POST',
        body: formData,
      });
      if (response.ok) {
        setSuccess(true);
      } else {
        alert('Error processing PDFs');
      }
    } catch (error) {
      console.error('Error:', error);
      alert('Error uploading files');
    }
    setProcessing(false);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-600 to-blue-500 text-white flex flex-col justify-center items-center p-4">
      <motion.div
        initial={{ opacity: 0, y: -50 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="text-center mb-8"
      >
        <h1 className="text-5xl font-bold mb-4">Create AI Assistant</h1>
        <p className="text-xl">Upload your documents to train your custom AI agent</p>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, scale: 0.5 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5, delay: 0.2 }}
        className="bg-white bg-opacity-20 p-8 rounded-lg shadow-lg w-full max-w-2xl"
      >
        <div className="flex flex-col items-center space-y-6">
          <div className="w-full">
            <label className="flex flex-col items-center px-4 py-6 border-2 border-white border-dashed rounded-lg cursor-pointer hover:bg-white hover:bg-opacity-10 transition-colors">
              <Upload size={48} />
              <span className="mt-2 text-lg">Drop PDFs here or click to upload</span>
              <input
                type="file"
                multiple
                accept=".pdf"
                onChange={handleFileChange}
                className="hidden"
              />
            </label>
          </div>

          {files.length > 0 && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="w-full"
            >
              <div className="bg-white bg-opacity-10 p-4 rounded-lg">
                <div className="flex items-center mb-2">
                  <FileText className="mr-2" />
                  <span className="font-semibold">Selected Files:</span>
                </div>
                <ul className="space-y-1">
                  {files.map((file, index) => (
                    <li key={index} className="text-sm">{file.name}</li>
                  ))}
                </ul>
              </div>
            </motion.div>
          )}

          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={handleUpload}
            disabled={processing || files.length === 0}
            className={`flex items-center justify-center px-8 py-3 rounded-full font-bold text-lg ${
              files.length === 0
                ? 'bg-gray-400 cursor-not-allowed'
                : 'bg-white text-purple-600 shadow-lg'
            }`}
          >
            {processing ? 'Processing...' : 'Process PDFs'}
            {!processing && <ArrowRight className="ml-2" />}
          </motion.button>
        </div>
      </motion.div>

      {success && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-8 text-center"
        >
          <div className="flex items-center justify-center mb-4">
            <CheckCircle size={32} className="text-green-400 mr-2" />
            <span className="text-xl font-semibold">PDFs processed successfully!</span>
          </div>
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => navigate('/query')}
            className="bg-green-500 text-white px-8 py-3 rounded-full font-bold text-lg shadow-lg flex items-center"
          >
            Go to Query Page
            <ArrowRight className="ml-2" />
          </motion.button>
        </motion.div>
      )}
    </div>
  );
};

export default UploadPage;