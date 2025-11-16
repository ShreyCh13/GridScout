## Smart Siting – Combined Backend & Frontend

This repository bundles:

- **backend**: Python node ranking engine + Flask API (serves ranking endpoints for the UI)
- **frontend**: React + Vite + Mapbox UI for exploring optimal siting locations

### Project Structure

- `backend/`
  - `api_server.py` – Flask API server (entry point)
  - `api_wrapper.py` – Translates frontend JSON into backend ranking calls
  - `node_ranking_engine.py` – Core ranking logic
  - `final_csv_v1_new.csv` – Node dataset used by the engine
  - `requirements.txt` – Python dependencies
  - `README.md` – Backend-specific documentation and details

- `frontend/`
  - `src/App.jsx` and other React components/styles
  - `public/` – Static assets (images, CSV used by the frontend)
  - `package.json`, `vite.config.js`, `tailwind.config.cjs`, `postcss.config.cjs`
  - `README.md` – Frontend-specific documentation

---

## Running the Backend (Flask API)

1. **Go to the backend folder:**

   ```bash
   cd backend
   ```

2. **Create and activate a virtual environment (optional but recommended):**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the API server:**

   ```bash
   python api_server.py
   ```

5. **Verify it is running:**

   - Health check: open `http://localhost:5001/api/health` in your browser or via curl.
   - Ranking endpoint: `POST http://localhost:5001/api/submit` (used by the React frontend).

The server expects `final_csv_v1_new.csv` to be present in the `backend/` folder (it is included in this repo).

---

## Running the Frontend (React + Vite)

1. **Open a new terminal and go to the frontend folder:**

   ```bash
   cd frontend
   ```

2. **Install dependencies:**

   ```bash
   npm install
   ```

3. **Start the dev server:**

   ```bash
   npm run dev
   ```

4. **Open the app in your browser:**

   - Vite will print a URL such as `http://localhost:5173/`.
   - Open that URL in your browser.

5. **Backend URL configuration:**

   - The frontend sends requests to the backend in `src/App.jsx` via a constant named `backendUrl`.
   - Ensure it is set to:

   ```js
   const backendUrl = 'http://localhost:5001/api/submit';
   ```

   - For deployment, replace this with your hosted backend URL (e.g. Render/Replit).

---

## Typical Local Workflow

1. **Backend:** run `python api_server.py` in `backend/`.
2. **Frontend:** run `npm run dev` in `frontend/`.
3. Use the React UI to configure loads/locations; it will call the Flask API and display ranked nodes on the map and in the dashboard.

---

## GitHub Usage

To publish this combined project on GitHub:

1. Create an empty public repository on GitHub (no README).
2. From the root of this project (`smart-siting-project`), run:

   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   git branch -M main
   git push -u origin main
   ```

3. Update `YOUR_USERNAME` and `YOUR_REPO_NAME` as appropriate.

You can then add a “Live demo” section linking to any hosted frontend/backend you deploy.


