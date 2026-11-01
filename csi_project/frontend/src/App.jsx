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

  // Overview statistics calculated from the loaded transition manifest
  const cleanRecords = transitionRecords.filter(
    (record) => record.scenario === 'clean'
  )

  const residualRecords = transitionRecords.filter(
    (record) => record.scenario === 'residual'
  )

  const totalPassed = transitionRecords.filter(
    (record) => record.label === 'pass'
  ).length

  const totalRejected = transitionRecords.filter(
    (record) => record.label === 'reject'
  ).length

  const carryoverRejections = transitionRecords.filter(
    (record) =>
      record.scenario === 'residual' &&
      record.phase === 'next_fill' &&
      record.label === 'reject'
  ).length

  const inspections = [
    {
      name: 'Clean transition',
      passed: cleanRecords.filter((record) => record.label === 'pass').length,
      rejected: cleanRecords.filter((record) => record.label === 'reject').length,
      total: cleanRecords.length,
      status: 'Review',
    },
    {
      name: 'Residual carryover',
      passed: residualRecords.filter((record) => record.label === 'pass').length,
      rejected: residualRecords.filter((record) => record.label === 'reject').length,
      total: residualRecords.length,
      status: 'Alert',
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
          Simulation dataset loaded
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
            SIMULATION DATA
          </div>
        </section>

        {/* OVERVIEW PAGE */}
        {page === 'Overview' && (
          <>
            <section className="metrics">
              <div className="metric-card">
                <span className="metric-label">TOTAL INSPECTIONS</span>
                <strong>{transitionLoading ? '—' : transitionRecords.length}</strong>
                <span className="metric-note">Loaded inspection records</span>
              </div>

              <div className="metric-card">
                <span className="metric-label">TOTAL PASSED</span>
                <strong className="green">
                  {transitionLoading ? '—' : totalPassed}
                </strong>
                <span className="metric-note">Proxy inspection labels</span>
              </div>

              <div className="metric-card">
                <span className="metric-label">TOTAL REJECTED</span>
                <strong className="red">
                  {transitionLoading ? '—' : totalRejected}
                </strong>
                <span className="metric-note">Proxy inspection labels</span>
              </div>

              <div className="metric-card">
                <span className="metric-label">CARRYOVER REJECTIONS</span>
                <strong className="orange">
                  {transitionLoading ? '—' : carryoverRejections}
                </strong>
                <span className="metric-note">
                  Rejected residual next-fill records
                </span>
              </div>
            </section>

            <section className="content-grid">
              <div className="panel">
                <div className="panel-heading">
                  <div>
                    <h3>Inspection Results</h3>
                    <p>Comparison by test scenario</p>
                  </div>
                  <span className="small-tag">
                    {transitionLoading
                      ? 'LOADING'
                      : `${transitionRecords.length} FRAMES`}
                  </span>
                </div>

                {inspections.map((item) => (
                  <div className="scenario" key={item.name}>
                    <div className="scenario-heading">
                      <strong>{item.name}</strong>
                      <span
                        className={item.status === 'Alert' ? 'alert' : 'review'}
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
                              ? (item.passed / item.total) * 100
                              : 0
                          }%`,
                        }}
                      />
                      <div
                        className="rejected-bar"
                        style={{
                          width: `${
                            item.total
                              ? (item.rejected / item.total) * 100
                              : 0
                          }%`,
                        }}
                      />
                    </div>

                    <div className="result-legend">
                      <span>
                        <i className="green-dot" />
                        {item.passed} Passed
                      </span>
                      <span>
                        <i className="red-dot" />
                        {item.rejected} Rejected
                      </span>
                    </div>
                  </div>
                ))}

                <p className="disclaimer">
                  Results are based on supplied transition-proxy labels,
                  not live production measurements.
                </p>
              </div>

              <div className="panel">
                <div className="panel-heading">
                  <div>
                    <h3>Station Status</h3>
                    <p>Current simulation overview</p>
                  </div>
                  <span className="online-tag">DATA AVAILABLE</span>
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
                  <strong>FillCam</strong>
                </div>
                <div className="station-detail">
                  <span>Previous recipe</span>
                  <strong>Recipe 5</strong>
                </div>
                <div className="station-detail">
                  <span>Next recipe</span>
                  <strong>Recipe 6</strong>
                </div>
                <div className="station-detail">
                  <span>Expected fill volume</span>
                  <strong>40 mL</strong>
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
                  <strong>Color carryover detected</strong>
                  <p>
                    Residual fluid was identified during the next-fill phase.
                  </p>
                </div>
                <span className="alert-level">HIGH</span>
              </div>

              <div className="alert-row">
                <span className="alert-icon warning">!</span>
                <div>
                  <strong>Color transition inconsistency</strong>
                  <p>
                    Observed color differs from the expected recipe color.
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
            transitionRecords={transitionRecords}
            transitionLoading={transitionLoading}
            transitionError={transitionError}
          />
        )}

        {/* RECIPE & CARRYOVER */}
        {page === 'Recipe & Carryover' && (
          <RecipeCarryover
            recipes={recipes}
            transitionRecords={transitionRecords}
          />
        )}

        {/* COPILOT ANALYSIS */}
        {page === 'Copilot Analysis' && (
          <CopilotAnalysis transitionRecords={transitionRecords} />
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
                <p>Selected inspection frames from transition 01</p>
              </div>
              <span className="small-tag">4 EXAMPLES</span>
            </div>

            <div className="history-list">
              {cameras.map((camera) => (
                <div className="history-row" key={camera.name}>
                  <div>
                    <strong>
                      {camera.name} · Frame {camera.frame}
                    </strong>
                    <p>
                      {camera.scenario} / {camera.phase}
                    </p>
                  </div>

                  <span
                    className={`result-pill ${camera.result.toLowerCase()}`}
                  >
                    {camera.result}
                  </span>

                  <button
                    className="text-button"
                    onClick={() => {
                      setPage('Cameras')
                      setSelectedCamera(camera)
                    }}
                  >
                    View frame
                  </button>
                </div>
              ))}
            </div>

            <p className="disclaimer">
              {cameras.length} representative camera examples are shown.
              The loaded transition manifest contains {transitionRecords.length} records.
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

function VisionInspection({
  transitionRecords,
  transitionLoading,
  transitionError,
}) {
  const [selectedIndex, setSelectedIndex] = useState(0)

  const selected = transitionRecords[selectedIndex]
  const imageName = selected?.image
    ? selected.image.split('/').pop()
    : ''

  const passed = transitionRecords.filter(
    (record) => record.label === 'pass'
  ).length

  const rejected = transitionRecords.filter(
    (record) => record.label === 'reject'
  ).length

  const phaseName = (phase) => {
    if (phase === 'previous_fill') return 'Previous recipe fill'
    if (phase === 'next_fill') return 'Next recipe fill'
    return phase || 'Unavailable'
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      <section className="metrics">
        <div className="metric-card">
          <span className="metric-label">INSPECTION FRAMES</span>
          <strong>{transitionLoading ? '—' : transitionRecords.length}</strong>
          <span className="metric-note">Saved simulation frames</span>
        </div>
        <div className="metric-card">
          <span className="metric-label">PASSED</span>
          <strong className="green">
            {transitionLoading ? '—' : passed}
          </strong>
          <span className="metric-note">Proxy labels</span>
        </div>
        <div className="metric-card">
          <span className="metric-label">REJECTED</span>
          <strong className="red">
            {transitionLoading ? '—' : rejected}
          </strong>
          <span className="metric-note">Proxy labels</span>
        </div>
        <div className="metric-card">
          <span className="metric-label">CAMERA STAGES</span>
          <strong>1 / 3</strong>
          <span className="metric-note">Evidence currently available</span>
        </div>
      </section>

      <section className="panel">
        <div className="panel-heading">
          <div>
            <h3>Multi-Stage Vision Inspection</h3>
            <p>Three inspection stages across the vial production process</p>
          </div>
          <span className="small-tag">SIMULATION EVIDENCE</span>
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
            <p className="metric-note">Saved transition inspection frames</p>
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
          Only vial-fill transition frames are currently connected.
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
                  ? `${selected.scenario || 'Scenario'} · ${phaseName(selected.phase)}`
                  : 'Select a record from the timeline'}
              </p>
            </div>
            {selected && (
              <span
                className={`result-pill ${
                  selected.label === 'reject' ? 'reject' : 'pass'
                }`}
              >
                {String(selected.label || 'unknown').toUpperCase()}
              </span>
            )}
          </div>

          {transitionLoading && (
            <p className="disclaimer">Loading inspection evidence...</p>
          )}
          {transitionError && (
            <p className="disclaimer">{transitionError}</p>
          )}

          {!transitionLoading && !transitionError && selected && (
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
                <img
                  src={`/data/images/${imageName}`}
                  alt="Selected vial inspection frame"
                  style={{
                    width: '100%',
                    maxHeight: 420,
                    objectFit: 'contain',
                  }}
                />
                <span className="frame-label">
                  FRAME {selected.frame ?? '—'}
                </span>
              </div>

              <div style={{ marginTop: 16 }}>
                <div className="station-detail">
                  <span>Scenario</span>
                  <strong>
                    {selected.scenario === 'residual'
                      ? 'Residual carryover'
                      : 'Clean transition'}
                  </strong>
                </div>
                <div className="station-detail">
                  <span>Inspection phase</span>
                  <strong>{phaseName(selected.phase)}</strong>
                </div>
                <div className="station-detail">
                  <span>Vial</span>
                  <strong>{selected.vial ?? '—'}</strong>
                </div>
                <div className="station-detail">
                  <span>Color error</span>
                  <strong>
                    {selected.color_error == null
                      ? '—'
                      : Number(selected.color_error).toFixed(3)}
                  </strong>
                </div>
                <div className="station-detail">
                  <span>Nozzle residual</span>
                  <strong
                    className={selected.nozzle_residual ? 'red' : 'green'}
                  >
                    {selected.nozzle_residual ? 'Detected' : 'Not detected'}
                  </strong>
                </div>
                <div className="station-detail">
                  <span>Expected volume</span>
                  <strong>
                    {selected.expected_volume_ml ?? '—'} mL
                  </strong>
                </div>
              </div>
            </>
          )}

          {!transitionLoading && !transitionError && !selected && (
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
              {transitionRecords.length} RECORDS
            </span>
          </div>

          {transitionLoading && (
            <p className="disclaimer">Loading timeline...</p>
          )}

          {!transitionLoading && !transitionError && (
            <div
              style={{
                display: 'flex',
                flexDirection: 'column',
                gap: 10,
                marginTop: 16,
              }}
            >
              {transitionRecords.map((record, index) => {
                const isReject = record.label === 'reject'
                const active = index === selectedIndex

                return (
                  <button
                    key={`${record.scenario}-${record.phase}-${index}`}
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
                        background: isReject ? '#f16b75' : '#43d6a0',
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
                        {record.scenario === 'residual'
                          ? 'Residual carryover'
                          : 'Clean transition'}
                      </strong>
                      <small style={{ color: '#91add0' }}>
                        {phaseName(record.phase)}
                      </small>
                      <small style={{ color: '#91add0' }}>
                        Vial {record.vial ?? '—'} · Frame {record.frame ?? '—'}
                      </small>
                    </span>

                    <span
                      className={`result-pill ${
                        isReject ? 'reject' : 'pass'
                      }`}
                    >
                      {String(record.label || 'unknown').toUpperCase()}
                    </span>
                  </button>
                )
              })}
            </div>
          )}
        </div>
      </section>

      <p className="disclaimer">
        These are representative simulation examples labeled transition_proxy.
        They are not live production decisions or verified machine-learning predictions.
      </p>
    </div>
  )
}

export default App