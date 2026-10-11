// Backend address used by index.html and admin.html.
// Local development: the backend runs separately on port 8000.
// Docker: nginx serves deploy/config.docker.js in place of this file, pointing at /api on the same origin.
window.CHATBOT_API_URL = "http://localhost:8000";
