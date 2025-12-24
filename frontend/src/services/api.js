import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const askQuestion = async (question, topK = 5) => {
  const response = await api.post('/ask', {
    question,
    top_k: topK,
  });
  return response.data;
};

export const generateRelatedWork = async (topic, numCitations = 5) => {
  const response = await api.post('/related-work', {
    topic,
    num_citations: numCitations,
  });
  return response.data;
};

export const getPapers = async () => {
  const response = await api.get('/papers');
  return response.data;
};

export const searchChunks = async (query, topK = 5) => {
  const response = await api.get('/search', {
    params: { query, top_k: topK },
  });
  return response.data;
};

export const checkHealth = async () => {
  const response = await api.get('/');
  return response.data;
};

export default api;