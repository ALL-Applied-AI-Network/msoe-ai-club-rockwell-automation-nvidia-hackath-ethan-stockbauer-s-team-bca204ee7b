
import { useMemo, useState } from 'react'

function colorString(rgb) {
  if (!Array.isArray(rgb) || rgb.length < 3) return null

  const normalized = Math.max(...rgb) <= 1

  const values = rgb.slice(0, 3).map((value) =>
    Math.max(
      0,
      Math.min(255, Math.round(normalized ? value * 255 : value))
    )
  )

  return `rgb(${values.join(', ')})`
}

function Metric({ label, value, detail }) {
  return (
    <div style={styles.metric}>
      <span style={styles.muted}>{label}</span>
      <strong style={styles.metricValue}>{value}</strong>
      <span style={styles.small}>{detail}</span>
    </div>
  )
}

function Detail({ label, value }) {
  return (
    <div style={styles.detailCard}>
      <span style={styles.muted}>{label}</span>
      <strong style={styles.detailValue}>{value}</strong>
    </div>
  )
}

export default function CopilotAnalysis({ transitionRecords = [] }) {
  const [scenario, setScenario] = useState('all')

  const records = useMemo(() => {
    return transitionRecords.filter(
      (record) =>
        scenario === 'all' || record.scenario === scenario
    )
  }, [transitionRecords, scenario])

  // Prefer the next-fill record so the analysis focuses on
  // the vial immediately following the recipe transition.
  const selectedRecord =
    records.find(
      (record) =>
        record.scenario === 'residual' &&
        record.phase === 'next_fill'
    ) ||
    records.find((record) => record.phase === 'next_fill') ||
    records[0]

  // Use the same label field as the other dashboard pages.
  const rejected = records.filter(
    (record) => record.label === 'reject'
  ).length

  const passed = records.filter(
    (record) => record.label === 'pass'
  ).length

  const previousRecipe = selectedRecord?.recipe_before ?? 5
  const nextRecipe = selectedRecord?.recipe_after ?? 6

  const expectedColor = colorString(
    selectedRecord?.expected_color_srgb
  )

  const observedColor = colorString(
    selectedRecord?.observed_color_srgb
  )

  const imagePath = selectedRecord?.image
    ? `/data/images/${selectedRecord.image.split('/').pop()}`
    : null

  const isRejected = selectedRecord?.label === 'reject'
  const isResidual = selectedRecord?.scenario === 'residual'

  return (
    <div style={styles.page}>
      {/* HEADER */}
      <div style={styles.header}>
        <div>
          <div style={styles.eyebrow}>
            MIXING COPILOT / INTELLIGENCE
          </div>
          <h2 style={styles.title}>Copilot Analysis</h2>
          <p style={styles.subtitle}>
            Recipe transition analysis and carryover inspection evidence
          </p>
        </div>

        <span style={styles.status}>SIMULATION DATA</span>
      </div>

      {/* DATA NOTICE */}
      <div style={styles.notice}>
        <strong>Analysis status</strong>
        <p>
          This page summarizes digital-twin simulation data. Rejection
          labels are transition proxies, not independently verified
          production defects. No validated AI prediction is currently
          connected.
        </p>
      </div>

      {/* METRICS */}
      <div style={styles.metrics}>
        <Metric
          label="Records analyzed"
          value={records.length}
          detail="Selected dataset records"
        />

        <Metric
          label="Proxy rejects"
          value={rejected}
          detail="Simulation labels"
        />

        <Metric
          label="Proxy passes"
          value={passed}
          detail="Simulation labels"
        />

        <Metric
          label="Recipe transition"
          value={
            selectedRecord
              ? `${previousRecipe} → ${nextRecipe}`
              : '—'
          }
          detail="Selected evidence record"
        />
      </div>

      {/* TRANSITION INVESTIGATION */}
      <section style={styles.panel}>
        <div style={styles.panelHeader}>
          <div>
            <h3 style={styles.heading}>
              Transition investigation
            </h3>
            <p style={styles.muted}>
              Compare clean and residual simulation scenarios
            </p>
          </div>

          <select
            value={scenario}
            onChange={(event) => setScenario(event.target.value)}
            style={styles.select}
          >
            <option value="all">All scenarios</option>
            <option value="clean">Clean</option>
            <option value="residual">Residual carryover</option>
          </select>
        </div>

        {selectedRecord ? (
          <>
            {/* COLOR COMPARISON */}
            <div style={styles.comparison}>
              <div style={styles.colorCard}>
                <span style={styles.muted}>
                  Expected recipe color
                </span>

                {expectedColor ? (
                  <div
                    style={{
                      ...styles.colorSwatch,
                      background: expectedColor,
                    }}
                  />
                ) : (
                  <div style={styles.unavailableSwatch}>
                    Color unavailable
                  </div>
                )}

                <span style={styles.small}>
                  Recipe {nextRecipe}
                </span>
              </div>

              <div style={styles.arrow}>→</div>

              <div style={styles.colorCard}>
                <span style={styles.muted}>
                  Observed simulated color
                </span>

                {observedColor ? (
                  <div
                    style={{
                      ...styles.colorSwatch,
                      background: observedColor,
                    }}
                  />
                ) : (
                  <div style={styles.unavailableSwatch}>
                    Color unavailable
                  </div>
                )}

                <span style={styles.small}>
                  {selectedRecord.scenario || 'No scenario selected'}
                </span>
              </div>
            </div>

            {/* RECORD DETAILS */}
            <div style={styles.recordDetails}>
              <Detail
                label="Color error"
                value={
                  selectedRecord.color_error == null
                    ? 'Unavailable'
                    : Number(selectedRecord.color_error).toFixed(4)
                }
              />

              <Detail
                label="Vial"
                value={selectedRecord.vial ?? '—'}
              />

              <Detail
                label="Transition position"
                value={
                  selectedRecord.vial_index_after_switch ?? '—'
                }
              />

              <div style={styles.detailCard}>
                <span style={styles.muted}>
                  Simulation result
                </span>

                <strong
                  style={{
                    ...styles.resultBadge,
                    color: isRejected ? '#ff8585' : '#69dfad',
                    background: isRejected
                      ? 'rgba(255,100,100,.12)'
                      : 'rgba(70,210,145,.12)',
                    borderColor: isRejected
                      ? '#a64c58'
                      : '#287a60',
                  }}
                >
                  {isRejected ? 'REJECT' : 'PASS'}
                </strong>
              </div>
            </div>

            {/* VISUAL EVIDENCE */}
            {imagePath && (
              <div style={styles.evidence}>
                <h4 style={styles.heading}>Visual evidence</h4>

                <div style={styles.imageContainer}>
                  <img
                    src={imagePath}
                    alt="Digital twin transition evidence"
                    style={styles.image}
                    onError={(event) => {
                      event.currentTarget.style.display = 'none'
                    }}
                  />
                </div>

                <span style={styles.small}>
                  Rendered digital-twin evidence — not a live camera feed
                </span>
              </div>
            )}
          </>
        ) : (
          <div style={styles.empty}>
            <strong>No transition records available</strong>
            <p style={styles.muted}>
              There are no records for the selected scenario.
            </p>
          </div>
        )}
      </section>

      {/* COPILOT EXPLANATION */}
      <section style={styles.panel}>
        <h3 style={styles.heading}>Copilot explanation</h3>

        {selectedRecord ? (
          <>
            <div style={styles.explanation}>
              <strong>
                {isResidual
                  ? 'Potential carryover during recipe changeover'
                  : 'Clean transition baseline'}
              </strong>

              {isResidual ? (
                <>
                  <p>
                    The simulated transition changes from Recipe{' '}
                    {previousRecipe} to Recipe {nextRecipe}. In the
                    residual scenario, the observed color is calculated
                    using a mixture of the previous recipe color and
                    the current recipe color.
                  </p>

                  <p>
                    This can cause a vial after a recipe change to
                    differ from its expected color.
                  </p>
                </>
              ) : (
                <>
                  <p>
                    This record represents the clean transition
                    scenario from Recipe {previousRecipe} to Recipe{' '}
                    {nextRecipe}.
                  </p>

                  <p>
                    Use this scenario as a baseline for comparing
                    the residual carryover examples.
                  </p>
                </>
              )}
            </div>

            <div style={styles.recommendation}>
              <strong>Suggested operator action</strong>

              <p>
                {isResidual
                  ? 'Review the first vial after the recipe change and inspect the shared fluid path for remaining material before continuing production.'
                  : 'Use the clean transition as a baseline and review the next-fill inspection evidence for consistency with the expected recipe color.'}
              </p>
            </div>
          </>
        ) : (
          <p style={styles.muted}>
            Load transition records to display an explanation.
          </p>
        )}
      </section>

      {/* DISCLAIMER */}
      <p style={styles.disclaimer}>
        Simulation-based analysis only. Color error is the renderer's
        RGB distance metric, not Delta E 2000. No validated model
        confidence score or production defect probability is available
        here. Missing color fields are not treated as measured values.
      </p>
    </div>
  )
}

const styles = {
  page: {
    display: 'flex',
    flexDirection: 'column',
    gap: 20,
    color: '#edf5ff',
    width: '100%',
    minWidth: 0,
  },

  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    gap: 16,
    flexWrap: 'wrap',
  },

  eyebrow: {
    color: '#48c8e8',
    fontSize: 10,
    letterSpacing: 1.5,
    fontWeight: 700,
  },

  title: {
    margin: '7px 0',
    color: '#ffffff',
    fontSize: 25,
  },

  subtitle: {
    margin: 0,
    color: '#a9c3e3',
    fontSize: 12,
    lineHeight: 1.6,
  },

  status: {
    color: '#70d8ee',
    background: 'rgba(32,184,213,.10)',
    border: '1px solid #246078',
    borderRadius: 20,
    padding: '8px 12px',
    fontSize: 10,
    fontWeight: 700,
    whiteSpace: 'nowrap',
  },

  notice: {
    padding: 16,
    background: '#182b42',
    border: '1px solid #31516f',
    borderRadius: 10,
    color: '#e7f2ff',
    fontSize: 12,
    lineHeight: 1.7,
  },

  metrics: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(165px, 1fr))',
    gap: 14,
  },

  metric: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'flex-start',
    gap: 9,
    minWidth: 0,
    padding: 18,
    background: '#101e32',
    border: '1px solid #263d59',
    borderRadius: 11,
  },

  metricValue: {
    color: '#ffffff',
    fontSize: 23,
    lineHeight: 1.35,
    overflowWrap: 'anywhere',
  },

  muted: {
    color: '#a9c3e3',
    fontSize: 11,
    lineHeight: 1.5,
  },

  small: {
    color: '#91a9c7',
    fontSize: 10,
    lineHeight: 1.5,
  },

  panel: {
    padding: 20,
    background: '#101e32',
    border: '1px solid #263d59',
    borderRadius: 11,
    minWidth: 0,
  },

  panelHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    gap: 14,
    flexWrap: 'wrap',
    marginBottom: 20,
  },

  heading: {
    margin: '0 0 7px',
    color: '#ffffff',
    fontSize: 14,
  },

  select: {
    padding: '10px 13px',
    borderRadius: 7,
    border: '1px solid #31516f',
    background: '#172a40',
    color: '#ffffff',
    fontSize: 12,
    cursor: 'pointer',
  },

  comparison: {
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    gap: 28,
    flexWrap: 'wrap',
    padding: '24px 12px',
  },

  colorCard: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    textAlign: 'center',
    gap: 11,
    minWidth: 150,
  },

  colorSwatch: {
    width: 120,
    height: 90,
    borderRadius: 11,
    border: '2px solid #506783',
    boxShadow: '0 5px 16px rgba(0,0,0,.2)',
  },

  unavailableSwatch: {
    width: 120,
    height: 90,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: 11,
    border: '1px dashed #506783',
    background: '#172a40',
    color: '#91a9c7',
    fontSize: 10,
    textAlign: 'center',
  },

  arrow: {
    color: '#48c8e8',
    fontSize: 30,
    fontWeight: 600,
  },

  recordDetails: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(135px, 1fr))',
    gap: 12,
    borderTop: '1px solid #29415d',
    paddingTop: 18,
  },

  detailCard: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'flex-start',
    justifyContent: 'center',
    gap: 9,
    padding: 14,
    background: '#172a40',
    border: '1px solid #29415d',
    borderRadius: 9,
    minWidth: 0,
  },

  detailValue: {
    color: '#ffffff',
    fontSize: 17,
    overflowWrap: 'anywhere',
  },

  resultBadge: {
    display: 'inline-block',
    border: '1px solid',
    borderRadius: 6,
    padding: '5px 10px',
    fontSize: 11,
    letterSpacing: 0.5,
  },

  evidence: {
    marginTop: 24,
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'flex-start',
    gap: 12,
  },

  imageContainer: {
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    width: '100%',
    minHeight: 100,
    padding: 12,
    background: '#0b1728',
    border: '1px solid #29415d',
    borderRadius: 10,
  },

  image: {
    display: 'block',
    width: '100%',
    maxWidth: 620,
    maxHeight: 420,
    borderRadius: 7,
    objectFit: 'contain',
  },

  explanation: {
    background: '#172a40',
    border: '1px solid #29415d',
    borderRadius: 9,
    padding: 17,
    color: '#e7f2ff',
    fontSize: 12,
    lineHeight: 1.8,
  },

  recommendation: {
    marginTop: 13,
    background: 'rgba(32,184,213,.08)',
    border: '1px solid #246078',
    borderRadius: 9,
    padding: 17,
    color: '#e7f2ff',
    fontSize: 12,
    lineHeight: 1.8,
  },

  empty: {
    padding: 24,
    textAlign: 'center',
    background: '#172a40',
    borderRadius: 9,
  },

  disclaimer: {
    color: '#91a9c7',
    fontSize: 10,
    lineHeight: 1.7,
    margin: 0,
  },
}
