import { useEffect, useState } from "react";
import "./App.css";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer
} from "recharts";

const API_URL =
  import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

const fraudExample = {
  Time: 406.0,
  V1: -2.3122265423263,
  V2: 1.95199201064158,
  V3: -1.60985073229769,
  V4: 3.9979055875468,
  V5: -0.522187864667764,
  V6: -1.42654531920595,
  V7: -2.53738730624579,
  V8: 1.39165724829804,
  V9: -2.77008927719433,
  V10: -2.77227214465915,
  V11: 3.20203320709635,
  V12: -2.89990738849473,
  V13: -0.595221881324605,
  V14: -4.28925378244217,
  V15: 0.389724120274487,
  V16: -1.14074717980657,
  V17: -2.83005567450437,
  V18: -0.0168224681808257,
  V19: 0.416955705037907,
  V20: 0.126910559061474,
  V21: 0.517232370861764,
  V22: -0.0350493686052974,
  V23: -0.465211076182388,
  V24: 0.320198198514526,
  V25: 0.0445191674731724,
  V26: 0.177839798284401,
  V27: 0.261145002567677,
  V28: -0.143275874698919,
  Amount: 0.0
};
const legitimateExample = {
  Time: 0,
  V1: -1.3598071336738,
  V2: -0.0727811733098497,
  V3: 2.53634673796914,
  V4: 1.37815522427443,
  V5: -0.338320769942518,
  V6: 0.462387777762292,
  V7: 0.239598554061257,
  V8: 0.0986979012610507,
  V9: 0.363786969611213,
  V10: 0.0907941719789316,
  V11: -0.551599533260813,
  V12: -0.617800855762348,
  V13: -0.991389847235408,
  V14: -0.311169353699879,
  V15: 1.46817697209427,
  V16: -0.470400525259478,
  V17: 0.207971241929242,
  V18: 0.0257905801985591,
  V19: 0.403992960255733,
  V20: 0.251412098239705,
  V21: -0.018306777944153,
  V22: 0.277837575558899,
  V23: -0.110473910188767,
  V24: 0.0669280749146731,
  V25: 0.128539358273528,
  V26: -0.189114843888824,
  V27: 0.133558376740387,
  V28: -0.0210530534538215,
  Amount: 149.62
};

function App() {
  const [predictions, setPredictions] = useState([]);
  const [transaction, setTransaction] = useState(fraudExample);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [stats, setStats] = useState(null);

  const loadPredictions = () => {
    fetch(`${API_URL}/predictions`)
      .then((response) => {
        if (!response.ok) {
          throw new Error("Failed to load predictions.");
        }

        return response.json();
      })
      .then((data) => {
        setPredictions(data);
      })
      .catch((error) => {
        setError(error.message);
      });
  };

  useEffect(() => {
    loadPredictions();
    loadStats();
  }, []);

  const loadStats = () => {
  fetch(`${API_URL}/stats`)
    .then((response) => {
      if (!response.ok) {
        throw new Error("Failed to load statistics.");
      }

      return response.json();
    })
    .then((data) => {
      setStats(data);
    })
    .catch((error) => {
      setError(error.message);
    });
};

  const handleChange = (event) => {
    const { name, value } = event.target;

    setTransaction({
      ...transaction,
      [name]: Number(value)
    });
  };

  const handleSubmit = async (event) => {
    event.preventDefault();

    setError("");
    setResult(null);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify(transaction)
      });

      if (!response.ok) {
        throw new Error("Prediction request failed.");
      }

      const data = await response.json();

      setResult(data);

      loadPredictions();
      loadStats();
    } catch (error) {
      setError(error.message);
    }
  };

  const chartData = stats
  ? [
      {
        name: "Fraud",
        count: stats.fraud_detected
      },
      {
        name: "Legitimate",
        count: stats.legitimate
      }
    ]
  : [];

  return (
    <div className="container">
      <h1>Fraud Detection Dashboard</h1>

      {stats && (
  <div className="stats-grid">
    <div className="stat-card">
      <h3>Total Predictions</h3>
      <p>{stats.total_predictions}</p>
    </div>

    <div className="stat-card">
      <h3>Fraud Detected</h3>
      <p>{stats.fraud_detected}</p>
    </div>

    <div className="stat-card">
      <h3>Legitimate</h3>
      <p>{stats.legitimate}</p>
    </div>

    <div className="stat-card">
      <h3>Average Risk Score</h3>
      <p>{(Number(stats.average_risk_score) * 100).toFixed(2)}%</p>
    </div>
  </div>
)}

    {stats && (
  <section className="card">
    <h2>Prediction Distribution</h2>

    <div className="chart-container">
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={chartData}>
          <XAxis dataKey="name" />
          <YAxis allowDecimals={false} />
          <Tooltip />
          <Bar dataKey="count" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  </section>
)}

      <section className="card">
        <h2>Analyze Transaction</h2>

        <p className="feature-note">
  V1–V28 are anonymized PCA-transformed features from the credit card
  transaction dataset. Time and Amount are the original transaction features.
</p>

        <p>
  Load an example transaction or manually modify the model features below.
</p>

<div className="example-buttons">
  <button
    type="button"
    onClick={() => setTransaction({ ...legitimateExample })}
  >
    Load Legitimate Example
  </button>

  <button
    type="button"
    onClick={() => setTransaction({ ...fraudExample })}
  >
    Load Fraud Example
  </button>
</div>

        <form onSubmit={handleSubmit}>
          <div className="form-grid">
            {Object.entries(transaction).map(([key, value]) => (
              <label key={key}>
                {key}

                <input
                  type="number"
                  step="any"
                  name={key}
                  value={value}
                  onChange={handleChange}
                />
              </label>
            ))}
          </div>

          <button type="submit">
            Analyze Transaction
          </button>
        </form>

        {result && (
  <div className={`result ${result.is_fraud ? "fraud-result" : "legitimate-result"}`}>
    <h3>Prediction Result</h3>

    <div className="result-details">
      <div>
        <span className="result-label">Risk Score</span>
        <strong>
          {(Number(result.risk_score) * 100).toFixed(2)}%
        </strong>
      </div>

      <div>
        <span className="result-label">Threshold</span>
        <strong>
          {(Number(result.threshold) * 100).toFixed(1)}%
        </strong>
      </div>

      <div>
        <span className="result-label">Classification</span>

        <span
          className={`badge ${
            result.is_fraud ? "fraud-badge" : "legitimate-badge"
          }`}
        >
          {result.is_fraud ? "Fraud" : "Legitimate"}
        </span>
      </div>
    </div>
  </div>
)}

        {error && <p>{error}</p>}
      </section>

      <section className="card">
        <h2>Recent Predictions</h2>

        <div className="table-container">
        <table>
          <thead>
            <tr>
              <th>ID</th>
              <th>Amount</th>
              <th>Risk Score</th>
              <th>Classification</th>
              <th>Threshold</th>
              <th>Created</th>
            </tr>
          </thead>

          <tbody>
            {predictions.map((prediction) => (
              <tr key={prediction.id}>
                <td>{prediction.id}</td>
                <td>${prediction.amount.toFixed(2)}</td>
                <td>
                  {(Number(prediction.risk_score) * 100).toFixed(2)}%
                </td>
                <td>
              <span
                 className={`badge ${
                  prediction.is_fraud ? "fraud-badge" : "legitimate-badge"
                      }`}
                    >
    {prediction.is_fraud ? "Fraud" : "Legitimate"}
  </span>
</td>
                <td>{prediction.threshold}</td>
                <td>{prediction.created_at}</td>
              </tr>
            ))}
          </tbody>
        </table>
        </div>
      </section>
    </div>
  );
}

export default App;