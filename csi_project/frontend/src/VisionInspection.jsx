
import { useEffect, useMemo, useState } from 'react'

function formatValue(value, digits = 3) {
  const number = Number(value)

  return value === undefined ||
    value === null ||
    value === '' ||
    !Number.isFinite(number)
    ? '—'
    : number.toFixed(digits)
}

export default function VisionInspection({
  transitionRecords = [],
  transitionLoading = false,
  transitionError = null,
}) {
  const records = transitionRecords
  const loading = transitionLoading
  const error = transitionError

  useEffect(() => {
    fetch('/data/transition_manifest.jsonl')
      .then((response) => {
        if (!response.ok) {
          throw new Error(`Manifest request failed: ${response.status}`)
        }
        return response.text()
      })
      .then((text) => {
        const parsed = text
          .split(/\r?\n/)
          .filter((line) => line.trim())
          .map((line) => JSON.parse(line))

        setRecords(parsed)
      })
      .catch((err) => {
        console.error('Could not load inspection records:', err)
        setError('Could not load the transition inspection manifest.')
      })
      .finally(() => setLoading(false))
  }, [])

  const selectedRecord = records[selectedIndex]

  const selectedImage = useMemo(() => {
    if (!selectedRecord?.image) return ''

    const filename = selectedRecord.image.split('/').pop()
    return `/data/images/${filename}`
  }, [selectedRecord])

  const passedCount = records.filter(
    (record) => record.label === 'pass'
  ).length

  const rejectedCount = records.filter(
    (record) => record.label === 'reject'
  ).length

  const phaseLabel = (phase) => {
    if (phase === 'previous_fill') return 'Previous recipe fill'
    if (phase === 'next_fill') return 'Next recipe fill'
    return phase || 'Inspection phase unavailable'
  }

  return (
    <div className="vision-page">
      {/* SUMMARY METRICS */}
      <section className="metrics">
        <div className="metric-card">
          <span className="metric-label">SAVED INSPECTION FRAMES</span>
          <strong>{loading ? '—' : records.length}</strong>
          <span className="metric-note">Loaded from transition manifest</span>
        </div>

        <div className="metric-card">
          <span className="metric-label">PASS LABELS</span>
          <strong className="green">{loading ? '—' : passedCount}</strong>
          <span className="metric-note">Transition-proxy labels</span>
        </div>

        <div className="metric-card">
          <span className="metric-label">REJECT LABELS</span>
          <strong className="red">{loading ? '—' : rejectedCount}</strong>
          <span className="metric-note">Transition-proxy labels</span>
        </div>

        <div className="metric-card">
          <span className="metric-label">CAMERA STAGES</span>
          <strong>1 / 3</strong>
          <span className="metric-note">Fill evidence currently available</span>
        </div>
      </section>

      {/* INSPECTION STAGES */}
      <section className="panel vision-panel">
        <div className="panel-heading">
          <div>
            <h3>Multi-Stage Vision Inspection</h3>
            <p>Inspection stages for one selected vial record</p>
          </div>
          <span className="small-tag">SAVED SIMULATION EVIDENCE</span>
        </div>

        <div className="vision-stage-grid">
          <div className="vision-stage-card selected">
            <div className="vision-stage-top">
              <span className="vision-stage-number">01</span>
              <span className="online-tag">EVIDENCE AVAILABLE</span>
            </div>
            <div className="vision-stage-icon">◉</div>
            <h3>Vial Fill</h3>
            <p>Saved transition inspection frame</p>
            <span className="vision-stage-status available">
              Image available
            </span>
          </div>

          <div className="vision-stage-card unavailable">
            <div className="vision-stage-top">
              <span className="vision-stage-number">02</span>
              <span className="small-tag">NOT CONNECTED</span>
            </div>
            <div className="vision-stage-icon">◌</div>
            <h3>Tank Level</h3>
            <p>Tank-level camera evidence</p>
            <span className="vision-stage-status pending">
              Awaiting source images
            </span>
          </div>

          <div className="vision-stage-card unavailable">
            <div className="vision-stage-top">
              <span className="vision-stage-number">03</span>
              <span className="small-tag">NOT CONNECTED</span>
            </div>
            <div className="vision-stage-icon">◇</div>
            <h3>Cap &amp; Seal</h3>
            <p>Finished vial and seal inspection</p>
            <span className="vision-stage-status pending">
              Awaiting source images
            </span>
          </div>
        </div>

        <p className="disclaimer">
          The three cards represent the intended inspection stages.
          Only vial-fill transition frames are currently connected;
          no live camera streams are displayed.
        </p>
      </section>

      {/* SELECTED EVIDENCE AND TIMELINE */}
      <section className="content-grid vision-content-grid">
        <div className="panel">
          <div className="panel-heading">
            <div>
              <h3>Selected Vial Evidence</h3>
              <p>
                {selectedRecord
                  ? `${selectedRecord.scenario || 'Scenario'} · ${phaseLabel(selectedRecord.phase)}`
                  : 'Choose an inspection record below'}
              </p>
            </div>

            {selectedRecord && (
              <span
                className={`result-pill ${
                  selectedRecord.label === 'reject' ? 'reject' : 'pass'
                }`}
              >
                {String(selectedRecord.label || 'unknown').toUpperCase()}
              </span>
            )}
          </div>

          {loading && (
            <p className="disclaimer">Loading inspection evidence...</p>
          )}

          {error && (
            <p className="disclaimer" role="alert">{error}</p>
          )}

          {!loading && !error && selectedRecord && (
            <>
              <div className="vision-image-wrap">
                <img
                  src={selectedImage}
                  alt={`Saved vial inspection frame ${selectedRecord.frame || ''}`}
                />
                <span className="frame-label">
                  FRAME {selectedRecord.frame ?? '—'}
                </span>
              </div>

              <div className="vision-record-details">
                <div className="station-detail">
                  <span>Scenario</span>
                  <strong>
                    {selectedRecord.scenario === 'residual'
                      ? 'Residual carryover'
                      : 'Clean transition'}
                  </strong>
                </div>

                <div className="station-detail">
                  <span>Inspection phase</span>
                  <strong>{phaseLabel(selectedRecord.phase)}</strong>
                </div>

                <div className="station-detail">
                  <span>Vial</span>
                  <strong>{selectedRecord.vial ?? '—'}</strong>
                </div>

                <div className="station-detail">
                  <span>Color error</span>
                  <strong>{formatValue(selectedRecord.color_error)}</strong>
                </div>

                <div className="station-detail">
                  <span>Nozzle residual</span>
                  <strong className={selectedRecord.nozzle_residual ? 'red' : 'green'}>
                    {selectedRecord.nozzle_residual
                      ? 'Detected'
                      : 'Not detected'}
                  </strong>
                </div>

                <div className="station-detail">
                  <span>Expected fill volume</span>
                  <strong>
                    {selectedRecord.expected_volume_ml ?? '—'} mL
                  </strong>
                </div>
              </div>
            </>
          )}

          {!loading && !error && !selectedRecord && (
            <p className="disclaimer">
              No inspection records were found in the manifest.
            </p>
          )}
        </div>

        {/* TIMELINE */}
        <div className="panel">
          <div className="panel-heading">
            <div>
              <h3>Inspection Timeline</h3>
              <p>Select a saved record to inspect its evidence</p>
            </div>
            <span className="small-tag">{records.length} RECORDS</span>
          </div>

          {loading && (
            <p className="disclaimer">Loading timeline...</p>
          )}

          {!loading && !error && records.length > 0 && (
            <div className="vision-timeline">
              {records.map((record, index) => {
                const isSelected = index === selectedIndex
                const isReject = record.label === 'reject'

                return (
                  <button
                    type="button"
                    key={`${record.scenario}-${record.phase}-${record.sequence_index ?? index}`}
                    className={`vision-timeline-item ${isSelected ? 'active' : ''}`}
                    onClick={() => setSelectedIndex(index)}
                  >
                    <span
                      className={`timeline-marker ${
                        isReject ? 'rejected' : 'passed'
                      }`}
                    />

                    <span className="timeline-copy">
                      <strong>
                        {record.scenario === 'residual'
                          ? 'Residual carryover'
                          : 'Clean transition'}
                      </strong>

                      <small>
                        {phaseLabel(record.phase)} · Vial {record.vial ?? '—'}
                      </small>

                      <small>Frame {record.frame ?? '—'}</small>
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

          {!loading && !error && records.length === 0 && (
            <p className="disclaimer">
              No timeline records are available.
            </p>
          )}
        </div>
      </section>

      <p className="disclaimer">
        These records are representative simulation examples labeled
        transition_proxy. Pass/reject labels are not live production
        decisions or verified machine-learning predictions.
      </p>
    </div>
  )
}

