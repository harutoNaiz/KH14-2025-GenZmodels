import React, { useState, useEffect } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import { motion } from 'framer-motion';
import { 
  ArrowLeft, 
  Send, 
  Download, 
  Trash2, 
  User, 
  Bot, 
  X,
  Database,
  FileText,
  Sparkles
} from 'lucide-react';

const ViewLLMPage = () => {
  const navigate = useNavigate();
  const { id } = useParams();
  const [query, setQuery] = useState('');
  const [conversations, setConversations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [loadingLLM, setLoadingLLM] = useState(true);
  const [llmName, setLLMName] = useState('');
  const [user, setUser] = useState(null);
  const [downloading, setDownloading] = useState(false);
  const [unlearningText, setUnlearningText] = useState('');
  const [showUnlearnModal, setShowUnlearnModal] = useState(false);
  const [unlearning, setUnlearning] = useState(false);

  useEffect(() => {
    const userStr = localStorage.getItem('user');
    if (!userStr) {
      navigate('/login');
      return;
    }
    const userObj = JSON.parse(userStr);
    setUser(userObj);
    loadLLM();
  }, [navigate, id]);

  const loadLLM = async () => {
    try {
      const response = await fetch(`http://localhost:5000/load-llm/${id}`);
      const data = await response.json();
      
      if (response.ok) {
        setLLMName(data.llmName);
      } else {
        alert('Error loading LLM: ' + data.error);
        navigate('/dashboard');
      }
    } catch (error) {
      console.error('Error:', error);
      alert('Error loading LLM');
      navigate('/dashboard');
    } finally {
      setLoadingLLM(false);
    }
  };

  const handleUnlearn = async () => {
    if (!unlearningText.trim()) return;
    
    setUnlearning(true);
    try {
      const response = await fetch('http://localhost:5000/unlearn-data', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: unlearningText,
          llmId: id,
          userEmail: user.email
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to unlearn data');
      }

      await loadLLM();
      setShowUnlearnModal(false);
      setUnlearningText('');
      alert('Successfully unlearned the specified data');
    } catch (error) {
      console.error('Error:', error);
      alert('Error unlearning data');
    } finally {
      setUnlearning(false);
    }
  };

  const handleBuildAgent = async () => {
    try {
      // Store current LLM ID in localStorage for use in agent building page
      localStorage.setItem('currentLLMId', id);
      // Navigate to the agent building page
      navigate(`/build-agent/${id}`);
    } catch (error) {
      console.error('Error:', error);
      alert('Error navigating to agent builder');
    }
  };

  const handleDownload = async (fileType) => {
    setDownloading(true);
    try {
      const response = await fetch(`http://localhost:5000/download-files/${fileType}`);
      
      if (!response.ok) {
        throw new Error('Download failed');
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = fileType === 'faiss' ? 'faiss_index.bin' : 'texts.pkl';
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
      console.error('Error downloading file:', error);
      alert('Error downloading file');
    } finally {
      setDownloading(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    try {
      const response = await fetch('http://localhost:5000/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          query,
          history: conversations 
        }),
      });

      if (response.ok) {
        const data = await response.json();
        setConversations(prev => [...prev, {
          question: query,
          answer: data.answer,
          timestamp: new Date().toISOString()
        }]);
        setQuery('');
      } else {
        setConversations(prev => [...prev, {
          question: query,
          answer: 'Error getting response',
          timestamp: new Date().toISOString()
        }]);
      }
    } catch (error) {
      console.error('Error:', error);
      setConversations(prev => [...prev, {
        question: query,
        answer: 'Error connecting to server',
        timestamp: new Date().toISOString()
      }]);
    }
    setLoading(false);
  };

  if (loadingLLM) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-purple-600 to-blue-500 text-white flex items-center justify-center">
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="text-center text-xl"
        >
          Loading your AI assistant...
        </motion.div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-600 to-blue-500 text-white">
      <div className="max-w-4xl mx-auto p-4">
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex justify-between items-center mb-8 pt-4"
        >
          <Link 
            to="/dashboard"
            className="flex items-center text-white hover:text-gray-200 transition-colors"
          >
            <ArrowLeft className="mr-2" />
            Back to Dashboard
          </Link>

          <div className="flex gap-3 flex-wrap justify-end">
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={handleBuildAgent}
              className="px-4 py-2 rounded-full bg-yellow-500 hover:bg-yellow-600 text-white flex items-center shadow-lg"
            >
              <Sparkles size={18} className="mr-2" />
              Build AI Agent
            </motion.button>
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => setShowUnlearnModal(true)}
              className="px-4 py-2 rounded-full bg-red-500 hover:bg-red-600 text-white flex items-center shadow-lg"
            >
              <Trash2 size={18} className="mr-2" />
              Unlearn Data
            </motion.button>
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => handleDownload('faiss')}
              disabled={downloading}
              className="px-4 py-2 rounded-full bg-green-500 hover:bg-green-600 text-white flex items-center shadow-lg"
            >
              <Database size={18} className="mr-2" />
              {downloading ? 'Downloading...' : 'FAISS Index'}
            </motion.button>
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => handleDownload('texts')}
              disabled={downloading}
              className="px-4 py-2 rounded-full bg-green-500 hover:bg-green-600 text-white flex items-center shadow-lg"
            >
              <FileText size={18} className="mr-2" />
              {downloading ? 'Downloading...' : 'Texts'}
            </motion.button>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="text-center mb-8"
        >
          <h1 className="text-4xl font-bold mb-4">{llmName}</h1>
          <p className="text-lg text-gray-100">Ask questions and get personalized answers</p>
        </motion.div>

        <motion.form 
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          onSubmit={handleSubmit} 
          className="mb-8"
        >
          <div className="flex gap-3">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Enter your question"
              className="flex-1 px-6 py-3 rounded-full bg-white bg-opacity-20 border border-white border-opacity-30 text-white placeholder-gray-300 focus:outline-none focus:ring-2 focus:ring-white focus:ring-opacity-50"
            />
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              type="submit"
              disabled={loading || !query.trim()}
              className={`px-8 py-3 rounded-full flex items-center ${
                loading || !query.trim() 
                  ? 'bg-gray-400 cursor-not-allowed' 
                  : 'bg-white text-purple-600 hover:bg-gray-100'
              } shadow-lg`}
            >
              {loading ? 'Thinking...' : 'Ask'}
              <Send className="ml-2" size={20} />
            </motion.button>
          </div>
        </motion.form>

        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3 }}
          className="bg-white bg-opacity-10 rounded-lg p-4 h-[600px] overflow-y-auto backdrop-blur-sm"
        >
          {conversations.length === 0 ? (
            <div className="text-center text-gray-300 py-10">
              No questions asked yet. Start by typing a question above!
            </div>
          ) : (
            conversations.map((conv, index) => (
              <motion.div 
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                key={index}
                className="bg-white bg-opacity-20 rounded-lg p-5 mb-4"
              >
                <div className="mb-4">
                  <div className="flex items-start gap-3">
                    <div className="w-8 h-8 bg-purple-500 rounded-full flex items-center justify-center">
                      <User size={16} />
                    </div>
                    <div>
                      <p className="text-white mb-1">{conv.question}</p>
                      <p className="text-sm text-gray-300">
                        {new Date(conv.timestamp).toLocaleString()}
                      </p>
                    </div>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <div className="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center">
                    <Bot size={16} />
                  </div>
                  <p className="text-white whitespace-pre-wrap">{conv.answer}</p>
                </div>
              </motion.div>
            ))
          )}
        </motion.div>
      </div>

      {/* Unlearn Modal */}
      {showUnlearnModal && (
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 backdrop-blur-sm"
        >
          <motion.div 
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            className="bg-white rounded-lg p-6 w-96 text-gray-800"
          >
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-xl font-semibold">Unlearn Data</h3>
              <button 
                onClick={() => setShowUnlearnModal(false)}
                className="text-gray-500 hover:text-gray-700"
              >
                <X size={20} />
              </button>
            </div>
            <textarea
              value={unlearningText}
              onChange={(e) => setUnlearningText(e.target.value)}
              placeholder="Enter the text you want to unlearn..."
              className="w-full h-32 p-4 rounded-lg border border-gray-300 mb-4 focus:outline-none focus:ring-2 focus:ring-purple-500"
            />
            <div className="flex justify-end gap-3">
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => setShowUnlearnModal(false)}
                className="px-4 py-2 rounded-full border border-gray-300 hover:bg-gray-100 transition-colors"
              >
                Cancel
              </motion.button>
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={handleUnlearn}
                disabled={unlearning || !unlearningText.trim()}
                className={`px-6 py-2 rounded-full text-white ${
                  unlearning || !unlearningText.trim()
                    ? 'bg-gray-400 cursor-not-allowed'
                    : 'bg-red-500 hover:bg-red-600'
                }`}
              >
                {unlearning ? 'Unlearning...' : 'Unlearn'}
              </motion.button>
            </div>
          </motion.div>
        </motion.div>
      )}
    </div>
  );
};

export default ViewLLMPage;