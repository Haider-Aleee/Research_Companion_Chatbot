import React, { useState, useEffect } from 'react';
import QuestionForm from './components/QuestionForm';
import AnswerDisplay from './components/AnswerDisplay';
import PapersList from './components/PapersList';
import RelatedWorkGenerator from './components/RelatedWorkGenerator';
import { askQuestion, checkHealth } from './services/api';
import './App.css';

function App() {
  const [activeTab, setActiveTab] = useState('qa');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [apiStatus, setApiStatus] = useState('checking');

  useEffect(() => {
    checkApiHealth();
  }, []);

  const checkApiHealth = async () => {
    try {
      const health = await checkHealth();
      setApiStatus('online');
      console.log('API Status:', health);
    } catch (err) {
      setApiStatus('offline');
      console.error('API offline:', err);
    }
  };

  const handleAskQuestion = async (question, topK) => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const data = await askQuestion(question, topK);
      setResult(data);
    } catch (err) {
      setError(err.message || 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="app-header">
        <h1>🎓 Research Paper QA System</h1>
        <p className="subtitle">
          Ask questions about research papers and get answers with citations
        </p>
        <div className={`status-indicator ${apiStatus}`}>
          <span className="status-dot"></span>
          API Status: {apiStatus}
        </div>
      </header>

      <nav className="tabs">
        <button
          className={activeTab === 'qa' ? 'active' : ''}
          onClick={() => setActiveTab('qa')}
        >
          💬 Q&A
        </button>
        <button
          className={activeTab === 'papers' ? 'active' : ''}
          onClick={() => setActiveTab('papers')}
        >
          📚 Papers
        </button>
        <button
          className={activeTab === 'related' ? 'active' : ''}
          onClick={() => setActiveTab('related')}
        >
          📝 Related Work
        </button>
      </nav>

      <main className="main-content">
        {apiStatus === 'offline' && (
          <div className="alert alert-error">
            ⚠️ Backend API is offline. Please make sure the FastAPI server is running.
            <button onClick={checkApiHealth} className="retry-button">
              🔄 Retry
            </button>
          </div>
        )}

        {activeTab === 'qa' && (
          <div className="tab-content">
            <QuestionForm onSubmit={handleAskQuestion} loading={loading} />
            {error && <div className="error-message">❌ Error: {error}</div>}
            <AnswerDisplay result={result} />
          </div>
        )}

        {activeTab === 'papers' && (
          <div className="tab-content">
            <PapersList />
          </div>
        )}

        {activeTab === 'related' && (
          <div className="tab-content">
            <RelatedWorkGenerator />
          </div>
        )}
      </main>

      <footer className="app-footer">
        <p>
          Built with ❤️ using RAG + LoRA Fine-tuning | 
          Powered by FLAN-T5 & Sentence Transformers
        </p>
      </footer>
    </div>
  );
}

export default App;