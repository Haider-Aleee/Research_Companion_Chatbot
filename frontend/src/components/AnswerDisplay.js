import React from 'react';
import ReactMarkdown from 'react-markdown';
import './AnswerDisplay.css';

const AnswerDisplay = ({ result }) => {
  if (!result) return null;

  return (
    <div className="answer-display">
      <div className="answer-section">
        <h2>💬 Answer</h2>
        <div className="answer-text">
          <ReactMarkdown>{result.answer}</ReactMarkdown>
        </div>
        <div className="metadata">
          <span>⏱️ Processing time: {result.processing_time.toFixed(2)}s</span>
        </div>
      </div>

      <div className="citations-section">
        <h3>📚 Citations</h3>
        <div className="citations-list">
          {result.citations.map((citation) => (
            <div key={citation.chunk_id} className="citation-card">
              <div className="citation-header">
                <span className="citation-index">[{citation.index}]</span>
                <span className="paper-id">{citation.paper_id}</span>
                <span className="relevance-badge">
                  {(citation.relevance_score * 100).toFixed(0)}% relevant
                </span>
              </div>
              <div className="citation-details">
                <span className="section-name">📄 Section: {citation.section}</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {result.retrieved_chunks && result.retrieved_chunks.length > 0 && (
        <div className="context-section">
          <h3>📖 Retrieved Context</h3>
          <details>
            <summary>Show retrieved text chunks ({result.retrieved_chunks.length})</summary>
            <div className="context-chunks">
              {result.retrieved_chunks.map((chunk, index) => (
                <div key={index} className="context-chunk">
                  <div className="chunk-header">
                    <strong>[{index + 1}]</strong> {chunk.paper_id} - {chunk.section}
                  </div>
                  <p className="chunk-text">{chunk.text}</p>
                </div>
              ))}
            </div>
          </details>
        </div>
      )}
    </div>
  );
};

export default AnswerDisplay;

