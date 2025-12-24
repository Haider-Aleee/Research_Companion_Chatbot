import React, { useState } from 'react';
import './QuestionForm.css';

const QuestionForm = ({ onSubmit, loading }) => {
  const [question, setQuestion] = useState('');
  const [topK, setTopK] = useState(5);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (question.trim()) {
      onSubmit(question, topK);
    }
  };

  const exampleQuestions = [
    "What are the main contributions of this paper?",
    "What methodology is used in this work?",
    "What are the baseline methods compared?",
    "What are the limitations mentioned?",
    "What datasets are used for evaluation?"
  ];

  return (
    <div className="question-form-container">
      <form onSubmit={handleSubmit} className="question-form">
        <div className="form-group">
          <label htmlFor="question">Ask a question about the research papers:</label>
          <textarea
            id="question"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="e.g., What are the main contributions of this paper?"
            rows={4}
            disabled={loading}
          />
        </div>

        <div className="form-row">
          <div className="form-group">
            <label htmlFor="topK">Number of sources (Top-K):</label>
            <input
              type="number"
              id="topK"
              value={topK}
              onChange={(e) => setTopK(parseInt(e.target.value))}
              min={1}
              max={10}
              disabled={loading}
            />
          </div>

          <button type="submit" disabled={loading || !question.trim()}>
            {loading ? '🔍 Searching...' : '🔍 Ask Question'}
          </button>
        </div>
      </form>

      <div className="example-questions">
        <p>Example questions:</p>
        <div className="example-buttons">
          {exampleQuestions.map((q, index) => (
            <button
              key={index}
              className="example-button"
              onClick={() => setQuestion(q)}
              disabled={loading}
            >
              {q}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

export default QuestionForm;