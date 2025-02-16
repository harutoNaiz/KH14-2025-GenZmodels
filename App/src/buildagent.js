import React, { useState, useEffect } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import { motion } from 'framer-motion';
import { 
  ArrowLeft, 
  Bot,
  Loader,
  AlertCircle
} from 'lucide-react';

const BuildAgentPage = () => {
  const navigate = useNavigate();
  const { id } = useParams();
  const [tasks, setTasks] = useState('');
  const [building, setBuilding] = useState(false);
  const [error, setError] = useState('');
  const [user, setUser] = useState(null);

  useEffect(() => {
    const userStr = localStorage.getItem('user');
    if (!userStr) {
      navigate('/login');
      return;
    }
    const userObj = JSON.parse(userStr);
    setUser(userObj);
  }, [navigate]);

  const handleBuild = async (e) => {
    e.preventDefault();
    if (!tasks.trim()) return;

    setBuilding(true);
    setError('');

    try {
      const response = await fetch('http://localhost:5000/build-agent', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          tasks,
          llmId: id,
          userEmail: user.email
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to build agent');
      }

      alert('AI Agent built successfully!');
      navigate(`/view-llm/${id}`);
    } catch (error) {
      console.error('Error:', error);
      setError(error.message || 'Error building AI agent');
    } finally {
      setBuilding(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-600 to-blue-500 text-white">
      <div className="max-w-4xl mx-auto p-4">
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center mb-8 pt-4"
        >
          <Link 
            to={`/llm/${id}`}
            className="flex items-center text-white hover:text-gray-200 transition-colors"
          >
            <ArrowLeft className="mr-2" />
            Back to LLM
          </Link>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="text-center mb-8"
        >
          <h1 className="text-4xl font-bold mb-4">Build AI Agent</h1>
          <p className="text-lg text-gray-100">Specify the tasks for your AI agent to perform</p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="bg-white bg-opacity-10 rounded-lg p-6 backdrop-blur-sm"
        >
          <form onSubmit={handleBuild}>
            <div className="mb-6">
              <label className="block text-white mb-2">Tasks Description</label>
              <textarea
                value={tasks}
                onChange={(e) => setTasks(e.target.value)}
                placeholder="Describe the tasks you want your AI agent to perform..."
                className="w-full h-48 p-4 rounded-lg bg-white bg-opacity-20 border border-white border-opacity-30 text-white placeholder-gray-300 focus:outline-none focus:ring-2 focus:ring-white focus:ring-opacity-50"
              />
            </div>

            {error && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="mb-4 p-4 bg-red-500 bg-opacity-20 rounded-lg flex items-center"
              >
                <AlertCircle className="mr-2" />
                {error}
              </motion.div>
            )}

            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              type="submit"
              disabled={building || !tasks.trim()}
              className={`w-full py-3 rounded-full flex items-center justify-center ${
                building || !tasks.trim()
                  ? 'bg-gray-400 cursor-not-allowed'
                  : 'bg-white text-purple-600 hover:bg-gray-100'
              } shadow-lg`}
            >
              {building ? (
                <>
                  <Loader className="animate-spin mr-2" size={20} />
                  Building Agent...
                </>
              ) : (
                <>
                  <Bot className="mr-2" size={20} />
                  Build Agent
                </>
              )}
            </motion.button>
          </form>
        </motion.div>
      </div>
    </div>
  );
};

export default BuildAgentPage;