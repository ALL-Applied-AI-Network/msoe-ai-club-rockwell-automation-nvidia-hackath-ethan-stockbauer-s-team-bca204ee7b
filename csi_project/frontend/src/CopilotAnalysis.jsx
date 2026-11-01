
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

export default function CopilotAnalysis({ inspections = [] }) {
  const records = useMemo(
    () => inspections.filter((record) => record.prediction),
    [inspections]
  )
  const [selectedVial, setSelectedVial] = useState('')
  const selectedRecord =
    records.find((record) => String(record.vial_id) === selectedVial) ||
    records[0]
  const prediction = selectedRecord?.prediction
  const previousRecipe = selectedRecord?.previous_recipe_id
  const nextRecipe = selectedRecord?.recipe_id
  const expectedColor = colorString(prediction?.expected_rgb)
  const observedColor = colorString(prediction?.observed_rgb)
  const imagePath = selectedRecord?.evidence?.[0]?.qa_url
  const lowConfidence = records.filter(
    (record) => record.measurement.status === 'low_confidence'
  ).length

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

        <span style={styles.status}>OPENCV RESULTS</span>
      </div>

      {/* DATA NOTICE */}
      <div style={styles.notice}>
        <strong>Analysis status</strong>
        <p>
          This page summarizes measured OpenCV results and descriptive color
          discrepancies. Expected colors are unvalidated proxies; no result is
          a verified contamination classification.
        </p>
      </div>

      {/* METRICS */}
      <div style={styles.metrics}>
        <Metric
          label="Inspected vials"
          value={inspections.length}
          detail="Exported dataset records"
        />

        <Metric
          label="Predictions available"
          value={records.length}
          detail="Descriptive comparisons"
        />

        <Metric
          label="Low confidence"
          value={lowConfidence}
          detail="Measurement quality"
        />

        <Metric
          label="Recipe transition"
          value={
            selectedRecord?.is_recipe_transition
              ? `${previousRecipe} → ${nextRecipe}`
              : selectedRecord
                ? `Recipe ${nextRecipe}`
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
              Inspect measured and expected color evidence
            </p>
          </div>

          <select
            value={selectedRecord ? String(selectedRecord.vial_id) : ''}
            onChange={(event) => setSelectedVial(event.target.value)}
            style={styles.select}
          >
            {records.map((record) => (
              <option key={record.vial_id} value={record.vial_id}>
                Vial {record.vial_id} · Recipe {record.recipe_id}
              </option>
            ))}
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
                  Observed OpenCV color
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
                  Vial {selectedRecord.vial_id}
                </span>
              </div>
            </div>

            {/* RECORD DETAILS */}
            <div style={styles.recordDetails}>
              <Detail
                label="Delta E2000"
                value={
                  prediction?.delta_e_2000 == null
                    ? 'Unavailable'
                    : Number(prediction.delta_e_2000).toFixed(2)
                }
              />

              <Detail
                label="Vial"
                value={selectedRecord.vial_id ?? '—'}
              />

              <Detail
                label="Mismatch index"
                value={
                  prediction?.mismatch_score == null
                    ? 'Unavailable'
                    : Number(prediction.mismatch_score).toFixed(3)
                }
              />

              <div style={styles.detailCard}>
                <span style={styles.muted}>
                  Measurement confidence
                </span>

                <strong
                  style={{
                    ...styles.resultBadge,
                    color: '#69dfad',
                    background: 'rgba(70,210,145,.12)',
                    borderColor: '#287a60',
                  }}
                >
                  {prediction?.measurement_confidence == null
                    ? 'Unavailable'
                    : `${(prediction.measurement_confidence * 100).toFixed(1)}%`}
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
                    alt="OpenCV color QA evidence"
                    style={styles.image}
                    onError={(event) => {
                      event.currentTarget.style.display = 'none'
                    }}
                  />
                </div>

                <span style={styles.small}>
                  Extracted frame with OpenCV QA overlay
                </span>
              </div>
            )}
          </>
        ) : (
          <div style={styles.empty}>
            <strong>No prediction records available</strong>
            <p style={styles.muted}>
              Run predictions and export the dashboard artifacts.
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
              <strong>Measured color discrepancy</strong>
              <p>{prediction?.explanation || 'No explanation is available.'}</p>
              <p>
                Expected color source:{' '}
                {prediction?.expected_color_source || 'Unavailable'}.
              </p>
            </div>

            <div style={styles.recommendation}>
              <strong>Suggested operator action</strong>

              <p>
                Review the source frame and QA overlay together. Treat the
                mismatch index as a screening aid, not a probability or a
                verified contamination decision.
              </p>
            </div>
          </>
        ) : (
          <p style={styles.muted}>
            Export prediction records to display an explanation.
          </p>
        )}
      </section>

      {/* DISCLAIMER */}
      <p style={styles.disclaimer}>
        Observed values are OpenCV measurements. Expected colors are
        unvalidated recipe proxies. Delta E2000 and mismatch indexes describe
        color difference; they do not establish contamination or defect status.
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
