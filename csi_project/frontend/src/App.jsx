import { useEffect, useState } from 'react'
import DigitalTwin from './DigitalTwin'
import RecipeCarryover from './RecipeCarryover'
import CopilotAnalysis from './CopilotAnalysis'
import './App.css'

const cameras = [
  {
    name: 'Camera 01',
    file: '000000_transition01_vial04_reject_frame0686.png',
    frame: '0686',
    phase: 'Previous fill',
    result: 'REJECT',
    scenario: 'Clean transition',
  },
  {
    name: 'Camera 02',
    file: '000001_transition01_vial04_pass_frame0021.png',
    frame: '0021',
    phase: 'Next fill',
    result: 'PASS',
    scenario: 'Clean transition',
  },
  {
    name: 'Camera 03',
    file: '000002_transition01_vial04_reject_frame0686.png',
    frame: '0686',
    phase: 'Previous fill',
    result: 'REJECT',
    scenario: 'Residual carryover',
  },
  {
    name: 'Camera 04',
    file: '000003_transition01_vial04_reject_frame0021.png',
    frame: '0021',
    phase: 'Next fill',
    result: 'REJECT',
    scenario: 'Residual carryover',
  },
]

function parseCSV(csv) {
  const lines = csv.trim().split(/\r?\n/)

  function splitRow(line) {
    const values = []
    let value = ''
    let inQuotes = false

    for (let i = 0; i < line.length; i++) {
      const char = line[i]

      if (char === '"') {
        if (inQuotes && line[i + 1] === '"') {
          value += '"'
          i++
        } else {
          inQuotes = !inQuotes
        }
      } else if (char === ',' && !inQuotes) {
        values.push(value.trim())
        value = ''
      } else {
        value += char
      }
    }

    values.push(value.trim())
    return values
  }

  if (!lines.length || !lines[0]) return []

  const headers = splitRow(lines[0]).map((header) =>
    header.replace(/^\uFEFF/, '').trim()
  )

  return lines.slice(1).filter((line) => line.trim()).map((line) => {
    const values = splitRow(line)
    return Object.fromEntries(
      headers.map((header, index) => [header, values[index] ?? ''])
    )
  })
}

function App() {
  const [page, setPage] = useState('Overview')
  const [selectedCamera, setSelectedCamera] = useState(null)

  const [recipes, setRecipes] = useState([])
  const [selectedRecipeId, setSelectedRecipeId] = useState('')
  const [recipeLoading, setRecipeLoading] = useState(true)
  const [recipeError, setRecipeError] = useState('')

  const [transitionRecords, setTransitionRecords] = useState([])
  const [transitionLoading, setTransitionLoading] = useState(true)
  const [transitionError, setTransitionError] = useState('')

  const [dashboardData, setDashboardData] = useState(null)
  const [dashboardLoading, setDashboardLoading] = useState(true)
  const [dashboardError, setDashboardError] = useState('')

  // Load recipe CSV
  useEffect(() => {
    fetch('/data/recipes.csv')
      .then((response) => {
        if (!response.ok) {
          throw new Error(`CSV request failed: ${response.status}`)
        }
        return response.text()
      })
      .then((csv) => {
        const rows = parseCSV(csv)

        const grouped = rows.reduce((groups, row) => {
          const id = String(row.recipe_id || '').trim()
          if (!id) return groups
          if (!groups[id]) groups[id] = []
          groups[id].push(row)
          return groups
        }, {})

        const recipeGroups = Object.entries(grouped)
          .sort(([a], [b]) => Number(a) - Number(b))
          .map(([id, vialRows]) => ({ id, vialRows }))

        setRecipes(recipeGroups)

        if (recipeGroups.length) {
          setSelectedRecipeId(recipeGroups[0].id)
        } else {
          setRecipeError(
            'The CSV loaded, but it did not contain any recipe_id records.'
          )
        }
      })
      .catch((error) => {
        console.error('Could not load recipes.csv:', error)
        setRecipeError(
          'Could not load /data/recipes.csv. Check that the file is in frontend/public/data/.'
        )
      })
      .finally(() => setRecipeLoading(false))
  }, [])

  // Load exported Mixing Copilot OpenCV results.
  useEffect(() => {
    fetch('/generated/mixing_copilot.json')
      .then((response) => {
        if (!response.ok) {
          throw new Error(`Dashboard export request failed: ${response.status}`)
        }
        return response.json()
      })
      .then((payload) => setDashboardData(payload))
      .catch((error) => {
        console.error('Could not load Mixing Copilot export:', error)
        setDashboardError(
          'Could not load generated Mixing Copilot results. Run the export-dashboard command.'
        )
      })
      .finally(() => setDashboardLoading(false))
  }, [])

  // Load transition manifest
  useEffect(() => {
    fetch('/data/transition_manifest.jsonl')
      .then((response) => {
        if (!response.ok) {
          throw new Error(`Manifest request failed: ${response.status}`)
        }
        return response.text()
      })
      .then((text) => {
        const records = text
          .split(/\r?\n/)
          .filter((line) => line.trim())
          .map((line) => JSON.parse(line))

        setTransitionRecords(records)
      })
      .catch((error) => {
        console.error('Could not load transition manifest:', error)
        setTransitionError(
          'Could not load /data/transition_manifest.jsonl. Check that the file is in frontend/public/data/.'
        )
      })
      .finally(() => setTransitionLoading(false))
  }, [])

  const selectedRecipe = recipes.find(
    (recipe) => recipe.id === selectedRecipeId
  )

  const selectedRecipeRow = selectedRecipe?.vialRows?.[0]
  const opencvInspections = dashboardData?.inspections || []
  const opencvSummary = dashboardData?.summary || {}
  const firstTransition = dashboardData?.transitions?.[0]

  const measurementResults = [
    {
      name: 'OpenCV frame measurements',
      accepted: opencvSummary.ok_frames || 0,
      review: opencvSummary.low_confidence_frames || 0,
      total: opencvSummary.frame_records || 0,
      status: 'MEASURED',
    },
    {
      name: 'Per-vial prediction coverage',
      accepted: opencvSummary.predictions_available || 0,
      review: Math.max(
        (opencvSummary.inspected_vials || 0) -
          (opencvSummary.predictions_available || 0),
        0
      ),
      total: opencvSummary.inspected_vials || 0,
      status: `${opencvSummary.predictions_available || 0} / ${opencvSummary.inspected_vials || 0} COVERAGE`,
      acceptedLabel: 'Predictions available',
      reviewLabel: 'Without predictions',
      neutralRemainder: true,
      note: `${opencvSummary.discrepancy_reviews || 0} descriptive discrepancy reviews · Rule-based, not verified defects`,
    },
  ]

  const navigation = [
    'Overview',
    'Digital Twin',
    'Vision Inspection',
    'Recipe & Carryover',
    'Copilot Analysis',
    'Cameras',
    'Recipes',
    'Transition Monitoring',
    'Inspection History',
  ]

  return (
    <div className="dashboard">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">CSI</div>
          <div>
            <h2>Fluid Systems</h2>
            <span>Inspection Platform</span>
          </div>
        </div>

        <p className="nav-label">WORKSPACE</p>

        {navigation.map((item) => (
          <button
            key={item}
            className={`nav-item ${page === item ? 'active' : ''}`}
            onClick={() => {
              setPage(item)
              setSelectedCamera(null)
            }}
          >
            {item === 'Overview' && '▦'}
            {item === 'Digital Twin' && '◇'}
            {item === 'Vision Inspection' && '◎'}
            {item === 'Recipe & Carryover' && '⇄'}
            {item === 'Copilot Analysis' && '✧'}
            {item === 'Cameras' && '◉'}
            {item === 'Recipes' && '◈'}
            {item === 'Transition Monitoring' && '⇄'}
            {item === 'Inspection History' && '✓'}
            <span>{item}</span>
          </button>
        ))}

        <div className="sidebar-bottom">
          <span className="status-dot"></span>
          {dashboardData
            ? 'OpenCV results loaded · Simulation available'
            : 'Simulation dataset loaded'}
        </div>
      </aside>

      <main className="main">
        <header className="topbar">
          <div>
            <p className="eyebrow">CONNECTED SYSTEMS INSTITUTE</p>
            <h1>Fluid Mixing Station 3</h1>
          </div>
          <div className="station-tag">STATION 03</div>
        </header>

        <section className="welcome">
          <div>
            <p className="eyebrow">INSPECTION DASHBOARD</p>
            <h2>{page}</h2>
            <p>
              Monitor vial filling, color consistency, and fluid carryover.
            </p>
          </div>
          <div className="simulation-badge">
            <span className="status-dot"></span>
            {dashboardData ? 'OPENCV RESULTS' : 'DATA LOADING'}
          </div>
        </section>

        {/* OVERVIEW PAGE */}
        {page === 'Overview' && (
          <>
            <section className="metrics">
              <div className="metric-card">
                <span className="metric-label">INSPECTED VIALS</span>
                <strong>{dashboardLoading ? '—' : opencvSummary.inspected_vials || 0}</strong>
                <span className="metric-note">Processed OpenCV records</span>
              </div>

              <div className="metric-card">
                <span className="metric-label">MEASURED FRAMES</span>
                <strong className="green">
                  {dashboardLoading ? '—' : opencvSummary.measured_frames || 0}
                </strong>
                <span className="metric-note">OpenCV color measurements</span>
              </div>

              <div className="metric-card">
                <span className="metric-label">LOW CONFIDENCE</span>
                <strong className="orange">
                  {dashboardLoading ? '—' : opencvSummary.low_confidence_frames || 0}
                </strong>
                <span className="metric-note">Measurements requiring review</span>
              </div>

              <div className="metric-card">
                <span className="metric-label">DISCREPANCY REVIEWS</span>
                <strong className="orange">
                  {dashboardLoading ? '—' : opencvSummary.discrepancy_reviews || 0}
                </strong>
                <span className="metric-note">
                  Descriptive Delta E results, not defects
                </span>
              </div>
            </section>

            <section className="content-grid">
              <div className="panel">
                <div className="panel-heading">
                  <div>
                    <h3>Inspection Results</h3>
                    <p>Measurement and discrepancy coverage</p>
                  </div>
                  <span className="small-tag">
                    {dashboardLoading
                      ? 'LOADING'
                      : `${opencvSummary.frame_records || 0} FRAMES`}
                  </span>
                </div>

                {measurementResults.map((item) => (
                  <div className="scenario" key={item.name}>
                    <div className="scenario-heading">
                      <strong>{item.name}</strong>
                      <span
                        className="review"
                      >
                        {item.status}
                      </span>
                    </div>

                    <div className="result-bar">
                      <div
                        className="passed-bar"
                        style={{
                          width: `${
                            item.total
                              ? (item.accepted / item.total) * 100
                              : 0
                          }%`,
                        }}
                      />
                      <div
                        className={item.neutralRemainder ? '' : 'rejected-bar'}
                        style={{
                          width: `${
                            item.total
                              ? (item.review / item.total) * 100
                              : 0
                          }%`,
                          ...(item.neutralRemainder
                            ? { height: '100%', background: '#52677f' }
                            : {}),
                        }}
                      />
                    </div>

                    <div className="result-legend">
                      <span>
                        <i className="green-dot" />
                        {item.accepted}{' '}
                        {item.acceptedLabel || 'Standard measurements'}
                      </span>
                      <span>
                        <i
                          className={item.neutralRemainder ? '' : 'red-dot'}
                          style={
                            item.neutralRemainder
                              ? {
                                  display: 'inline-block',
                                  width: 8,
                                  height: 8,
                                  borderRadius: '50%',
                                  marginRight: 6,
                                  verticalAlign: 'middle',
                                  background: '#52677f',
                                }
                              : undefined
                          }
                        />
                        {item.review} {item.reviewLabel || 'Review records'}
                      </span>
                    </div>
                    {item.note && (
                      <p className="disclaimer" style={{ marginTop: 10 }}>
                        {item.note}
                      </p>
                    )}
                  </div>
                ))}

                <p className="disclaimer">
                  Results come from the exported OpenCV pipeline. Review counts
                  are descriptive quality checks, not contamination decisions.
                </p>
              </div>

              <div className="panel">
                <div className="panel-heading">
                  <div>
                    <h3>Station Status</h3>
                    <p>Current exported pipeline result</p>
                  </div>
                  <span className="online-tag">
                    {dashboardError ? 'DATA UNAVAILABLE' : 'OPENCV DATA'}
                  </span>
                </div>

                <div className="station-visual">
                  <div className="station-circle">03</div>
                  <div>
                    <h2>Fluid Mixing Station</h2>
                    <p>Recipe transition inspection</p>
                  </div>
                </div>

                <div className="station-detail">
                  <span>Camera source</span>
                  <strong>{opencvInspections[0]?.camera_id || '—'}</strong>
                </div>
                <div className="station-detail">
                  <span>Previous recipe</span>
                  <strong>
                    {firstTransition ? `Recipe ${firstTransition.from_recipe_id}` : '—'}
                  </strong>
                </div>
                <div className="station-detail">
                  <span>Next recipe</span>
                  <strong>
                    {firstTransition ? `Recipe ${firstTransition.to_recipe_id}` : '—'}
                  </strong>
                </div>
                <div className="station-detail">
                  <span>Generated</span>
                  <strong>{dashboardData?.generated_at_utc || '—'}</strong>
                </div>
              </div>
            </section>

            <section className="panel alerts-panel">
              <div className="panel-heading">
                <div>
                  <h3>Quality Alerts</h3>
                  <p>Issues identified in the transition dataset</p>
                </div>
              </div>

              <div className="alert-row">
                <span className="alert-icon">!</span>
                <div>
                  <strong>Rule-based discrepancy review</strong>
                  <p>
                    {opencvSummary.discrepancy_reviews || 0} prediction records
                    contain descriptive Delta E comparisons.
                  </p>
                </div>
                <span className="alert-level warning-text">REVIEW</span>
              </div>

              <div className="alert-row">
                <span className="alert-icon warning">!</span>
                <div>
                  <strong>Measurement quality review</strong>
                  <p>
                    {opencvSummary.low_confidence_frames || 0} frames have low
                    measurement confidence.
                  </p>
                </div>
                <span className="alert-level warning-text">REVIEW</span>
              </div>
            </section>
          </>
        )}

        {/* DIGITAL TWIN */}
        {page === 'Digital Twin' && <DigitalTwin />}

        {/* VISION INSPECTION */}
        {page === 'Vision Inspection' && (
          <VisionInspection
            inspections={opencvInspections}
            loading={dashboardLoading}
            error={dashboardError}
          />
        )}

        {/* RECIPE & CARRYOVER */}
        {page === 'Recipe & Carryover' && (
          <RecipeCarryover
            recipes={recipes}
            inspections={opencvInspections}
          />
        )}

        {/* COPILOT ANALYSIS */}
        {page === 'Copilot Analysis' && (
          <CopilotAnalysis inspections={opencvInspections} />
        )}

        {/* CAMERAS */}
        {page === 'Cameras' && (
          <section className="panel">
            <div className="panel-heading">
              <div>
                <h3>Camera Monitoring</h3>
                <p>Simulation inspection frames from FillCam</p>
              </div>
              <span className="online-tag">4 FRAMES LOADED</span>
            </div>

            <div className="camera-grid">
              {cameras.map((camera) => (
                <button
                  className="camera-card"
                  key={camera.name}
                  onClick={() => setSelectedCamera(camera)}
                >
                  <div className="camera-image">
                    <img
                      src={`/cameras/${camera.file}`}
                      alt={camera.name}
                    />
                    <span className="frame-label">
                      FRAME {camera.frame}
                    </span>
                  </div>

                  <div className="camera-info">
                    <div>
                      <strong>{camera.name}</strong>
                      <p>
                        {camera.scenario} · {camera.phase}
                      </p>
                    </div>
                    <span
                      className={`result-pill ${camera.result.toLowerCase()}`}
                    >
                      {camera.result}
                    </span>
                  </div>

                  <span className="view-frame">Click to inspect frame →</span>
                </button>
              ))}
            </div>

            {selectedCamera && (
              <div className="selected-frame">
                <button
                  className="close-button"
                  onClick={() => setSelectedCamera(null)}
                >
                  Close ×
                </button>

                <h3>
                  {selectedCamera.name} · Frame {selectedCamera.frame}
                </h3>

                <img
                  src={`/cameras/${selectedCamera.file}`}
                  alt="Selected inspection frame"
                />

                <p>
                  {selectedCamera.scenario} / {selectedCamera.phase} /
                  <strong> {selectedCamera.result}</strong>
                </p>
              </div>
            )}

            <p className="disclaimer">
              These are saved simulation frames, not live camera streams.
              Labels are supplied transition-proxy inspection labels.
            </p>
          </section>
        )}

        {/* RECIPES */}
        {page === 'Recipes' && (
          <>
            <section className="panel recipe-library-panel">
              <div className="panel-heading">
                <div>
                  <h3>Recipe Library</h3>
                  <p>Recipe configurations loaded from the source CSV.</p>
                </div>
                <span className="online-tag">
                  {recipeLoading
                    ? 'LOADING DATA'
                    : `${recipes.length} RECIPES LOADED`}
                </span>
              </div>

              {recipeLoading && (
                <p className="disclaimer">Loading recipe data...</p>
              )}

              {!recipeLoading && recipeError && (
                <p className="disclaimer" role="alert">
                  {recipeError}
                </p>
              )}

              {!recipeLoading && !recipeError && recipes.length > 0 && (
                <>
                  <div className="recipe-picker-grid">
                    {recipes.map((recipe) => {
                      const row = recipe.vialRows[0]
                      const active = recipe.id === selectedRecipeId

                      return (
                        <button
                          type="button"
                          key={recipe.id}
                          className={`recipe-picker-card ${
                            active ? 'active' : ''
                          }`}
                          onClick={() => setSelectedRecipeId(recipe.id)}
                        >
                          <span className="recipe-picker-label">
                            RECIPE {recipe.id}
                          </span>
                          <strong>{row.type || 'Recipe'}</strong>
                          <span className="recipe-picker-volume">
                            {row.total_ml || '—'} mL
                          </span>
                          <small>
                            {row.color_purpose || 'No color description'}
                          </small>
                          <small>
                            {recipe.vialRows.length} vial records
                          </small>
                        </button>
                      )
                    })}
                  </div>

                  {selectedRecipeRow && (
                    <div className="recipe-selected-details">
                      <div className="panel-heading">
                        <div>
                          <h3>Recipe {selectedRecipeId} Details</h3>
                          <p>Values shown directly from the source CSV.</p>
                        </div>
                        <span className="small-tag">SOURCE CSV</span>
                      </div>

                      <div className="content-grid recipe-detail-grid">
                        <div className="recipe-detail-box">
                          <span>RECIPE TYPE</span>
                          <strong>{selectedRecipeRow.type || '—'}</strong>
                        </div>
                        <div className="recipe-detail-box">
                          <span>TOTAL VOLUME</span>
                          <strong>
                            {selectedRecipeRow.total_ml || '—'} mL
                          </strong>
                        </div>
                        <div className="recipe-detail-box">
                          <span>DISPENSE ORDER</span>
                          <strong>
                            {selectedRecipeRow.dispense_order || '—'}
                          </strong>
                        </div>
                      </div>

                      <h3 className="recipe-section-title">
                        Ingredient Volumes
                      </h3>

                      <div className="ingredient-volume-grid">
                        {[
                          ['T100', selectedRecipeRow.t100_ml],
                          ['T200', selectedRecipeRow.t200_ml],
                          ['T400', selectedRecipeRow.t400_ml],
                        ].map(([name, amount]) => (
                          <div
                            className="ingredient-volume-card"
                            key={name}
                          >
                            <span>{name}</span>
                            <strong>
                              {amount === '' || amount == null ? '—' : amount}
                              {' '}
                              <small>mL</small>
                            </strong>
                          </div>
                        ))}
                      </div>

                      <div className="recipe-purpose-box">
                        <span>COLOR PURPOSE / DESCRIPTION</span>
                        <strong>
                          {selectedRecipeRow.color_purpose || 'Not provided'}
                        </strong>
                        <p>
                          This is the source dataset's description, not a
                          measured color value.
                        </p>
                      </div>

                      <h3 className="recipe-section-title">Vial Records</h3>

                      <div className="recipe-table-wrap">
                        <table className="recipe-data-table">
                          <thead>
                            <tr>
                              <th>Vial</th>
                              <th>Start Time</th>
                              <th>End Time</th>
                              <th>Duration (s)</th>
                              <th>Start Frame</th>
                              <th>End Frame</th>
                            </tr>
                          </thead>
                          <tbody>
                            {selectedRecipe.vialRows.map((row, index) => (
                              <tr key={`${row.vial || 'vial'}-${index}`}>
                                <td>{row.vial || '—'}</td>
                                <td>{row.start_time || '—'}</td>
                                <td>{row.end_time || '—'}</td>
                                <td>{row.duration_s || '—'}</td>
                                <td>{row.start_frame || '—'}</td>
                                <td>{row.end_frame || '—'}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}
                </>
              )}
            </section>

            <p className="disclaimer">
              Recipe records are read from the local CSV file. Camera and
              inspection results elsewhere on this dashboard remain saved
              simulation samples.
            </p>
          </>
        )}

        {/* TRANSITION MONITORING */}
        {page === 'Transition Monitoring' && (
          <section className="panel">
            <div className="panel-heading">
              <div>
                <h3>Recipe Transition: 5 → 6</h3>
                <p>Inspection records from the transition simulation.</p>
              </div>
              <span className="online-tag">
                {transitionLoading
                  ? 'LOADING'
                  : transitionError
                    ? 'DATA UNAVAILABLE'
                    : `${transitionRecords.length} RECORDS LOADED`}
              </span>
            </div>

            {transitionLoading && (
              <p className="disclaimer">Loading inspection records...</p>
            )}

            {transitionError && (
              <p className="disclaimer" role="alert">
                {transitionError}
              </p>
            )}

            {!transitionLoading &&
              !transitionError &&
              transitionRecords.length > 0 && (
                <>
                  <div className="metrics">
                    <div className="metric-card">
                      <span className="metric-label">TOTAL RECORDS</span>
                      <strong>{transitionRecords.length}</strong>
                      <span className="metric-note">Visible test examples</span>
                    </div>

                    <div className="metric-card">
                      <span className="metric-label">PASSED</span>
                      <strong className="green">
                        {
                          transitionRecords.filter(
                            (record) => record.label === 'pass'
                          ).length
                        }
                      </strong>
                      <span className="metric-note">
                        Proxy inspection labels
                      </span>
                    </div>

                    <div className="metric-card">
                      <span className="metric-label">REJECTED</span>
                      <strong className="red">
                        {
                          transitionRecords.filter(
                            (record) => record.label === 'reject'
                          ).length
                        }
                      </strong>
                      <span className="metric-note">
                        Proxy inspection labels
                      </span>
                    </div>

                    <div className="metric-card">
                      <span className="metric-label">NOZZLE RESIDUAL</span>
                      <strong className="orange">
                        {
                          transitionRecords.filter(
                            (record) => record.nozzle_residual
                          ).length
                        }
                      </strong>
                      <span className="metric-note">
                        Residual detections
                      </span>
                    </div>
                  </div>

                  <div className="content-grid" style={{ marginTop: 22 }}>
                    {['clean', 'residual'].map((scenario) => {
                      const records = transitionRecords.filter(
                        (record) => record.scenario === scenario
                      )
                      const passed = records.filter(
                        (record) => record.label === 'pass'
                      ).length
                      const rejected = records.filter(
                        (record) => record.label === 'reject'
                      ).length
                      const total = records.length || 1

                      return (
                        <div className="panel" key={scenario}>
                          <div className="panel-heading">
                            <div>
                              <h3>
                                {scenario === 'clean'
                                  ? 'Clean Transition'
                                  : 'Residual Carryover'}
                              </h3>
                              <p>
                                {scenario === 'clean'
                                  ? 'Expected color after recipe change'
                                  : 'Nozzle residue remains during next fill'}
                              </p>
                            </div>
                            <span
                              className={
                                scenario === 'clean' ? 'review' : 'alert'
                              }
                            >
                              {scenario === 'clean' ? 'BASELINE' : 'ALERT'}
                            </span>
                          </div>

                          <div className="station-detail">
                            <span>Records</span>
                            <strong>{records.length}</strong>
                          </div>
                          <div className="station-detail">
                            <span>Passed</span>
                            <strong className="green">{passed}</strong>
                          </div>
                          <div className="station-detail">
                            <span>Rejected</span>
                            <strong className="red">{rejected}</strong>
                          </div>

                          <div
                            className="result-bar"
                            style={{ marginTop: 16 }}
                          >
                            <div
                              className="passed-bar"
                              style={{
                                width: `${(passed / total) * 100}%`,
                              }}
                            />
                            <div
                              className="rejected-bar"
                              style={{
                                width: `${(rejected / total) * 100}%`,
                              }}
                            />
                          </div>
                        </div>
                      )
                    })}
                  </div>

                  <h3 className="recipe-section-title">Inspection Frames</h3>

                  <div className="camera-grid">
                    {transitionRecords.map((record, index) => {
                      const imageName = (record.image || '').split('/').pop()
                      const imageUrl = `/data/images/${imageName}`
                      const isReject = record.label === 'reject'

                      return (
                        <div
                          className="camera-card"
                          key={`${record.scenario}-${record.sequence_index}-${index}`}
                        >
                          <div className="camera-image">
                            <img
                              src={imageUrl}
                              alt={`Transition ${record.label} frame ${record.frame}`}
                            />
                            <span className="frame-label">
                              FRAME {record.frame}
                            </span>
                          </div>

                          <div className="camera-info">
                            <div>
                              <strong>
                                {record.scenario === 'clean'
                                  ? 'Clean transition'
                                  : 'Residual carryover'}
                              </strong>
                              <p>
                                {record.phase === 'previous_fill'
                                  ? 'Previous fill'
                                  : 'Next fill'}{' '}
                                · Vial {record.vial}
                              </p>
                            </div>

                            <span
                              className={`result-pill ${
                                isReject ? 'reject' : 'pass'
                              }`}
                            >
                              {String(record.label).toUpperCase()}
                            </span>
                          </div>

                          <div style={{ padding: '0 14px 14px' }}>
                            <div className="station-detail">
                              <span>Color error</span>
                              <strong>
                                {Number(record.color_error || 0).toFixed(3)}
                              </strong>
                            </div>
                            <div className="station-detail">
                              <span>Nozzle residual</span>
                              <strong
                                className={
                                  record.nozzle_residual ? 'red' : 'green'
                                }
                              >
                                {record.nozzle_residual
                                  ? 'Detected'
                                  : 'Not detected'}
                              </strong>
                            </div>
                            <div className="station-detail">
                              <span>Expected volume</span>
                              <strong>
                                {record.expected_volume_ml ?? '—'} mL
                              </strong>
                            </div>
                          </div>
                        </div>
                      )
                    })}
                  </div>
                </>
              )}

            {!transitionLoading &&
              !transitionError &&
              transitionRecords.length === 0 && (
                <p className="disclaimer">
                  The manifest loaded, but it contains no inspection records.
                </p>
              )}

            <p className="disclaimer">
              These are representative simulation examples labeled
              transition_proxy. They are not live production measurements
              or live machine-learning predictions.
            </p>
          </section>
        )}

        {/* INSPECTION HISTORY */}
        {page === 'Inspection History' && (
          <section className="panel">
            <div className="panel-heading">
              <div>
                <h3>Inspection History</h3>
                <p>Processed vial records from the Mixing Copilot export</p>
              </div>
              <span className="small-tag">{opencvInspections.length} VIALS</span>
            </div>

            <div className="history-list">
              {opencvInspections.map((inspection) => (
                <div className="history-row" key={inspection.vial_id}>
                  <div>
                    <strong>
                      Vial {inspection.vial_id} · Recipe {inspection.recipe_id}
                    </strong>
                    <p>
                      {inspection.measurement.status.replace('_', ' ')} ·{' '}
                      {inspection.measurement.frame_count} measured stages
                    </p>
                  </div>

                  <span
                    className={`result-pill ${
                      inspection.measurement.status === 'ok' ? 'pass' : 'reject'
                    }`}
                  >
                    {inspection.measurement.status === 'ok'
                      ? 'MEASURED'
                      : 'REVIEW'}
                  </span>

                  <button
                    className="text-button"
                    onClick={() => setPage('Vision Inspection')}
                  >
                    View evidence
                  </button>
                </div>
              ))}
            </div>

            <p className="disclaimer">
              These are OpenCV measurement records. A review badge indicates
              measurement quality or discrepancy evidence, not a verified defect.
            </p>
          </section>
        )}

        <footer>
          CSI FLUID MIXING STATION 3 <span>•</span> HACKATHON PROJECT 2026
        </footer>
      </main>
    </div>
  )
}

function VisionInspection({ inspections, loading, error }) {
  const [selectedIndex, setSelectedIndex] = useState(0)

  const selected = inspections[selectedIndex]
  const selectedEvidence = selected?.evidence?.[0]
  const measured = inspections.filter(
    (record) => record.measurement.status === 'ok'
  ).length
  const review = inspections.length - measured

  const stageName = (stage) =>
    stage ? stage.replaceAll('_', ' ') : 'Unavailable'

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      <section className="metrics">
        <div className="metric-card">
          <span className="metric-label">INSPECTED VIALS</span>
          <strong>{loading ? '—' : inspections.length}</strong>
          <span className="metric-note">Processed pipeline records</span>
        </div>
        <div className="metric-card">
          <span className="metric-label">MEASURED</span>
          <strong className="green">
            {loading ? '—' : measured}
          </strong>
          <span className="metric-note">Standard measurement status</span>
        </div>
        <div className="metric-card">
          <span className="metric-label">REVIEW</span>
          <strong className="orange">
            {loading ? '—' : review}
          </strong>
          <span className="metric-note">Low or missing confidence</span>
        </div>
        <div className="metric-card">
          <span className="metric-label">EVIDENCE IMAGES</span>
          <strong>
            {inspections.reduce((count, item) => count + item.evidence.length, 0)}
          </strong>
          <span className="metric-note">Extracted and QA pairs</span>
        </div>
      </section>

      <section className="panel">
        <div className="panel-heading">
          <div>
            <h3>Multi-Stage Vision Inspection</h3>
            <p>Three inspection stages across the vial production process</p>
          </div>
          <span className="small-tag">OPENCV EVIDENCE</span>
        </div>

        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
            gap: 14,
            marginTop: 20,
          }}
        >
          <div
            style={{
              background: '#122d46',
              border: '1px solid #19b9f5',
              borderRadius: 10,
              padding: 22,
              textAlign: 'center',
            }}
          >
            <span className="online-tag">AVAILABLE</span>
            <div style={{ fontSize: 36, color: '#27baff', margin: 15 }}>
              ◉
            </div>
            <h3>Vial Fill</h3>
            <p className="metric-note">Extracted and measured vial frames</p>
            <span className="green">Image evidence available</span>
          </div>

          <div
            style={{
              background: '#102238',
              border: '1px dashed #52677f',
              borderRadius: 10,
              padding: 22,
              textAlign: 'center',
              opacity: 0.7,
            }}
          >
            <span className="small-tag">NOT CONNECTED</span>
            <div style={{ fontSize: 36, color: '#91add0', margin: 15 }}>
              ◌
            </div>
            <h3>Tank Level</h3>
            <p className="metric-note">Tank-level camera evidence</p>
            <span>Awaiting source images</span>
          </div>

          <div
            style={{
              background: '#102238',
              border: '1px dashed #52677f',
              borderRadius: 10,
              padding: 22,
              textAlign: 'center',
              opacity: 0.7,
            }}
          >
            <span className="small-tag">NOT CONNECTED</span>
            <div style={{ fontSize: 36, color: '#91add0', margin: 15 }}>
              ◇
            </div>
            <h3>Cap &amp; Seal</h3>
            <p className="metric-note">Finished vial and seal inspection</p>
            <span>Awaiting source images</span>
          </div>
        </div>

        <p className="disclaimer">
          Only vial-fill OpenCV frames are currently connected.
          Tank-level and cap/seal views are placeholders, not live camera feeds.
        </p>
      </section>

      <section className="content-grid" style={{ alignItems: 'start' }}>
        <div className="panel">
          <div className="panel-heading">
            <div>
              <h3>Selected Vial Evidence</h3>
              <p>
                {selected
                  ? `Vial ${selected.vial_id} · Recipe ${selected.recipe_id}`
                  : 'Select a record from the timeline'}
              </p>
            </div>
            {selected && (
              <span
                className={`result-pill ${
                  selected.measurement.status === 'ok' ? 'pass' : 'reject'
                }`}
              >
                {selected.measurement.status === 'ok' ? 'MEASURED' : 'REVIEW'}
              </span>
            )}
          </div>

          {loading && (
            <p className="disclaimer">Loading inspection evidence...</p>
          )}
          {error && (
            <p className="disclaimer">{error}</p>
          )}

          {!loading && !error && selected && (
            <>
              <div
                style={{
                  position: 'relative',
                  marginTop: 16,
                  minHeight: 240,
                  background: '#071321',
                  border: '1px solid #29445e',
                  borderRadius: 9,
                  display: 'flex',
                  justifyContent: 'center',
                  alignItems: 'center',
                  overflow: 'hidden',
                }}
              >
                {selectedEvidence?.qa_url ? (
                  <img
                    src={selectedEvidence.qa_url}
                    alt="Selected vial OpenCV QA evidence"
                    style={{
                      width: '100%',
                      maxHeight: 420,
                      objectFit: 'contain',
                    }}
                  />
                ) : (
                  <p className="disclaimer">QA image unavailable</p>
                )}
                <span className="frame-label">
                  FRAME {selectedEvidence?.decoded_frame_index ?? '—'}
                </span>
              </div>

              <div style={{ marginTop: 16 }}>
                <div className="station-detail">
                  <span>Measurement status</span>
                  <strong>{selected.measurement.status.replace('_', ' ')}</strong>
                </div>
                <div className="station-detail">
                  <span>Inspection stage</span>
                  <strong>{stageName(selectedEvidence?.stage)}</strong>
                </div>
                <div className="station-detail">
                  <span>Vial</span>
                  <strong>{selected.vial_id ?? '—'}</strong>
                </div>
                <div className="station-detail">
                  <span>Observed RGB</span>
                  <strong>
                    {selectedEvidence?.observed_rgb == null
                      ? '—'
                      : selectedEvidence.observed_rgb.join(', ')}
                  </strong>
                </div>
                <div className="station-detail">
                  <span>Measurement confidence</span>
                  <strong>
                    {selectedEvidence?.confidence == null
                      ? '—'
                      : `${(selectedEvidence.confidence * 100).toFixed(1)}%`}
                  </strong>
                </div>
                <div className="station-detail">
                  <span>Alignment confidence</span>
                  <strong>
                    {selected.measurement.alignment_confidence == null
                      ? '—'
                      : `${(
                          selected.measurement.alignment_confidence * 100
                        ).toFixed(1)}%`}
                  </strong>
                </div>
              </div>
            </>
          )}

          {!loading && !error && !selected && (
            <p className="disclaimer">No inspection records found.</p>
          )}
        </div>

        <div className="panel">
          <div className="panel-heading">
            <div>
              <h3>Inspection Timeline</h3>
              <p>Select a frame to inspect its evidence</p>
            </div>
            <span className="small-tag">
              {inspections.length} VIALS
            </span>
          </div>

          {loading && (
            <p className="disclaimer">Loading timeline...</p>
          )}

          {!loading && !error && (
            <div
              style={{
                display: 'flex',
                flexDirection: 'column',
                gap: 10,
                marginTop: 16,
              }}
            >
              {inspections.map((record, index) => {
                const needsReview = record.measurement.status !== 'ok'
                const active = index === selectedIndex

                return (
                  <button
                    key={record.vial_id}
                    onClick={() => setSelectedIndex(index)}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 12,
                      width: '100%',
                      padding: 14,
                      border: `1px solid ${
                        active ? '#19b9f5' : '#263f59'
                      }`,
                      borderRadius: 8,
                      background: active ? '#15314b' : '#102238',
                      color: '#eaf3ff',
                      textAlign: 'left',
                      cursor: 'pointer',
                    }}
                  >
                    <span
                      style={{
                        width: 10,
                        height: 10,
                        flexShrink: 0,
                        borderRadius: '50%',
                        background: needsReview ? '#f2a15d' : '#43d6a0',
                      }}
                    />

                    <span
                      style={{
                        display: 'flex',
                        flex: 1,
                        flexDirection: 'column',
                        gap: 4,
                      }}
                    >
                      <strong style={{ fontSize: 12 }}>
                        Vial {record.vial_id} · Recipe {record.recipe_id}
                      </strong>
                      <small style={{ color: '#91add0' }}>
                        {record.measurement.frame_count} stage records
                      </small>
                      <small style={{ color: '#91add0' }}>
                        Confidence{' '}
                        {record.measurement.confidence_median == null
                          ? '—'
                          : `${(
                              record.measurement.confidence_median * 100
                            ).toFixed(1)}%`}
                      </small>
                    </span>

                    <span
                      className={`result-pill ${
                        needsReview ? 'reject' : 'pass'
                      }`}
                    >
                      {needsReview ? 'REVIEW' : 'MEASURED'}
                    </span>
                  </button>
                )
              })}
            </div>
          )}
        </div>
      </section>

      <p className="disclaimer">
        These are measured OpenCV results. Confidence describes measurement
        quality; it is not a defect or contamination probability.
      </p>
    </div>
  )
}

export default App
