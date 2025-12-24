import React, { useEffect, useState } from 'react';
import { getPapers } from '../services/api';
import './PapersList.css';

const PapersList = () => {
  const [papers, setPapers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadPapers();
  }, []);

  const loadPapers = async () => {
    try {
      const data = await getPapers();
      setPapers(data);
      setLoading(false);
    } catch (err) {
      setError(err.message);
      setLoading(false);
    }
  };

  if (loading) return <div className="loading">Loading papers...</div>;
  if (error) return <div className="error">Error: {error}</div>;

  return (
    <div className="papers-list">
      <h2>📚 Available Papers ({papers.length})</h2>
      <div className="papers-grid">
        {papers.map((paper) => (
          <div key={paper.paper_id} className="paper-card">
            <h3>{paper.paper_id}</h3>
            <div className="paper-stats">
              <span>📑 {paper.num_sections} sections</span>
              <span>📦 {paper.num_chunks} chunks</span>
            </div>
            <details className="paper-sections">
              <summary>View sections</summary>
              <ul>
                {paper.sections.map((section, idx) => (
                  <li key={idx}>{section}</li>
                ))}
              </ul>
            </details>
          </div>
        ))}
      </div>
    </div>
  );
};

export default PapersList;