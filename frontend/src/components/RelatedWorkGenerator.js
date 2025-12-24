import React, { useState } from 'react';
import { generateRelatedWork } from '../services/api';
import './RelatedWorkGenerator.css';

const RelatedWorkGenerator = () => {
  const [topic, setTopic] = useState('');
  const [numCitations, setNumCitations] = useState(5);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleGenerate = async (e) => {
    e.preventDefault();
    if (!topic.trim()) return;

    setLoading(true);
    setError(null);

    try {
      const data = await generateRelatedWork(topic, numCitations);
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="related-work-generator">
      <h2>📝 Related Work Generator</h2>
      <form onSubmit={handleGenerate} className="generator-form">
        <div className="form-group">
          <label htmlFor="topic">Topic:</label>
          <input
            type="text"
            id="topic"
            value={topic}
            onChange={(e) => setTopic(e.target.value)}
            placeholder="e.g., transformer architectures, attention mechanisms"
            disabled={loading}
          />
        </div>

        <div className="form-group">
          <label htmlFor="numCitations">Number of citations:</label>
          <input
            type="number"
            id="numCitations"
            value={numCitations}
            onChange={(e) => setNumCitations(parseInt(e.target.value))}
            min={3}
            max={10}
            disabled={loading}
          />
        </div>

        <button type="submit" disabled={loading || !topic.trim()}>
          {loading ? '✍️ Generating...' : '✍️ Generate Related Work'}
        </button>
      </form>

      {error && <div className="error">{error}</div>}

      {result && (
        <div className="result-display">
          <h3>Generated Paragraph:</h3>
          <div className="paragraph-box">
            <p>{result.paragraph}</p>
          </div>

          <h3>Citations:</h3>
          <ul className="citations-list">
            {result.citations.map((citation) => (
              <li key={citation.index}>
                [{citation.index}] {citation.paper_id}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default RelatedWorkGenerator;

