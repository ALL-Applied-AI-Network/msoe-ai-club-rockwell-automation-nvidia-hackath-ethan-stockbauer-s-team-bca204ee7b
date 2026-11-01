
import { useMemo, useState } from 'react'

function RecipeCarryover({ recipes = [], transitionRecords = [] }) {
  const [previousId, setPreviousId] = useState('5')
  const [currentId, setCurrentId] = useState('6')
  const [scenario, setScenario] = useState('residual')

  const previousRecipe = recipes.find(
    (recipe) => recipe.id === previousId
  )

  const currentRecipe = recipes.find(
    (recipe) => recipe.id === currentId
  )

  const previousRow = previousRecipe?.vialRows?.[0]
  const currentRow = currentRecipe?.vialRows?.[0]

  // The available inspection manifest represents Recipe 5 → Recipe 6.
  const evidenceMatchesSelection =
    previousId === '5' && currentId === '6'

  const evidence = useMemo(
    () =>
      transitionRecords.filter(
        (record) => record.scenario === scenario
      ),
    [transitionRecords, scenario]
  )

  const previousFillEvidence = evidence.find(
    (record) => record.phase === 'previous_fill'
  )

  const nextFillEvidence = evidence.find(
    (record) => record.phase === 'next_fill'
  )

  const formatAmount = (value) =>
    value === undefined || value === null || value === ''
      ? '—'
      : `${value} mL`

  const getIngredients = (row) => {
    if (!row) return []

    return [
      { name: 'T-100', amount: row.t100_ml, color: '#d9e8f3' },
      { name: 'T-200', amount: row.t200_ml, color: '#f28b43' },
      { name: 'T-400', amount: row.t400_ml, color: '#398bd6' },
    ]
  }

  const previousIngredients = getIngredients(previousRow)
  const currentIngredients = getIngredients(currentRow)

  return (
    <div style={styles.page}>

      {/* RECIPE TRANSITION HEADER */}
      <section className="panel">
        <div className="panel-heading">
          <div>
            <h3>Recipe Transition Analysis</h3>
            <p>Compare the outgoing and incoming fluid recipes</p>
          </div>
          <span className="small-tag">RECIPE COMPARISON</span>
        </div>

        <div style={styles.transitionGrid}>
          <div style={styles.previousCard}>
            <span style={styles.label}>PREVIOUS RECIPE</span>

            <select
              value={previousId}
              onChange={(event) => setPreviousId(event.target.value)}
              style={styles.select}
            >
              {recipes.map((recipe) => (
                <option key={recipe.id} value={recipe.id}>
                  Recipe {recipe.id}
                </option>
              ))}
            </select>

            <h2 style={styles.recipeTitle}>
              Recipe {previousId}
            </h2>

            <p style={styles.description}>
              {previousRow?.color_purpose ||
                'Recipe description unavailable'}
            </p>

            <span style={styles.outgoingBadge}>
              OUTGOING FLUID
            </span>
          </div>

          <div style={styles.arrow}>→</div>

          <div style={styles.currentCard}>
            <span style={styles.label}>INCOMING RECIPE</span>

            <select
              value={currentId}
              onChange={(event) => setCurrentId(event.target.value)}
              style={styles.select}
            >
              {recipes.map((recipe) => (
                <option key={recipe.id} value={recipe.id}>
                  Recipe {recipe.id}
                </option>
              ))}
            </select>

            <h2 style={styles.recipeTitle}>
              Recipe {currentId}
            </h2>

            <p style={styles.description}>
              {currentRow?.color_purpose ||
                'Recipe description unavailable'}
            </p>

            <span style={styles.incomingText}>NEXT FLUID</span>
          </div>
        </div>
      </section>

      {/* INGREDIENT COMPARISON */}
      <section className="content-grid">
        <IngredientCard
          title="Previous Recipe Ingredients"
          recipeId={previousId}
          row={previousRow}
          ingredients={previousIngredients}
          badge="OUTGOING"
        />

        <IngredientCard
          title="Incoming Recipe Ingredients"
          recipeId={currentId}
          row={currentRow}
          ingredients={currentIngredients}
          badge="INCOMING"
        />
      </section>

      {/* SHARED FLUID PATH */}
      <section className="panel">
        <div className="panel-heading">
          <div>
            <h3>Shared Fluid Path</h3>
            <p>Why recipe order matters during a transition</p>
          </div>
          <span className="small-tag">PROCESS OVERVIEW</span>
        </div>

        <div style={styles.processFlow}>
          {[
            ['TANKS', 'Fluid supply'],
            ['MIXING PATH', 'Shared pathway'],
            ['NOZZLE', 'Filling outlet'],
            ['VIAL', 'New product'],
          ].map(([name, description], index) => (
            <div key={name} style={styles.processItem}>
              <div
                style={{
                  ...styles.processBox,
                  borderColor: index === 1 ? '#f28b43' : '#29445e',
                }}
              >
                <strong
                  style={{
                    color: index === 1 ? '#f2a15d' : '#f4f8ff',
                  }}
                >
                  {name}
                </strong>
                <p style={styles.processDescription}>
                  {description}
                </p>
              </div>

              {index !== 3 && (
                <span style={styles.arrowSmall}>→</span>
              )}
            </div>
          ))}
        </div>

        <div style={styles.warningBox}>
          <strong style={{ color: '#ffad65' }}>
            Potential carryover point
          </strong>
          <p style={styles.warningText}>
            Because recipes share a fluid pathway, residual fluid
            from the previous recipe may affect the initial output
            of the incoming recipe. The transition simulation
            provides evidence of this behavior in selected test records.
          </p>
        </div>
      </section>

      {/* INSPECTION EVIDENCE */}
      <section className="panel">
        <div className="panel-heading">
          <div>
            <h3>Transition Inspection Evidence</h3>
            <p>
              Review saved simulation examples for the available
              Recipe 5 → Recipe 6 transition
            </p>
          </div>

          <select
            value={scenario}
            onChange={(event) => setScenario(event.target.value)}
            style={{ ...styles.select, maxWidth: 190 }}
          >
            <option value="residual">Residual carryover</option>
            <option value="clean">Clean transition</option>
          </select>
        </div>

        {/* Make the evidence source explicit */}
        <div style={styles.evidenceSource}>
          <strong>Evidence source: Recipe 5 → Recipe 6</strong>
          <span>
            The inspection manifest currently contains this transition only.
          </span>
        </div>

        {!evidenceMatchesSelection && (
          <div style={styles.mismatchNotice}>
            <strong>Different recipe selection</strong>
            <p>
              You are comparing Recipe {previousId} → Recipe {currentId},
              but the inspection images and labels below still belong
              to the Recipe 5 → Recipe 6 simulation. They are shown
              for reference and are not evidence for your selected pair.
            </p>
          </div>
        )}

        {transitionRecords.length === 0 && (
          <p className="disclaimer">
            No transition evidence is currently available.
          </p>
        )}

        {transitionRecords.length > 0 && evidence.length === 0 && (
          <p className="disclaimer">
            No inspection records are available for this scenario.
          </p>
        )}

        {evidence.length > 0 && (
          <div className="content-grid" style={{ marginTop: 18 }}>
            {[previousFillEvidence, nextFillEvidence].map(
              (record, index) => (
                <div key={index} style={styles.evidenceCard}>
                  <span style={styles.label}>
                    {index === 0 ? 'PREVIOUS FILL' : 'NEXT FILL'}
                  </span>

                  {record ? (
                    <>
                      <div style={styles.evidenceImage}>
                        <img
                          src={`/data/images/${record.image?.split('/').pop()}`}
                          alt={`${scenario} ${record.phase} inspection`}
                          style={styles.image}
                        />
                      </div>

                      <div className="station-detail">
                        <span>Frame</span>
                        <strong>{record.frame ?? '—'}</strong>
                      </div>

                      <div className="station-detail">
                        <span>Inspection label</span>
                        <strong
                          className={
                            record.label === 'reject' ? 'red' : 'green'
                          }
                        >
                          {String(record.label || 'unknown').toUpperCase()}
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
                        <span>Color error</span>
                        <strong>
                          {record.color_error == null
                            ? '—'
                            : Number(record.color_error).toFixed(3)}
                        </strong>
                      </div>
                    </>
                  ) : (
                    <p className="disclaimer">
                      No {index === 0 ? 'previous-fill' : 'next-fill'}
                      record is available for this scenario.
                    </p>
                  )}
                </div>
              )
            )}
          </div>
        )}

        {scenario === 'residual' &&
          nextFillEvidence?.nozzle_residual && (
            <div style={styles.alertBox}>
              <strong style={{ color: '#ff8b7d' }}>
                Residual fluid detected in selected next-fill record
              </strong>
              <p style={styles.warningText}>
                The selected simulation record identifies nozzle
                residual during the next-fill phase. This is
                evidence from the simulation, not a live production alert.
              </p>
            </div>
          )}

        <p className="disclaimer">
          The displayed images and labels are representative simulation
          evidence. They do not establish a live contamination probability
          or a verified production defect.
        </p>
      </section>
    </div>
  )
}

function IngredientCard({
  title,
  recipeId,
  row,
  ingredients,
  badge,
}) {
  return (
    <div className="panel">
      <div className="panel-heading">
        <div>
          <h3>{title}</h3>
          <p>Recipe {recipeId} · Source CSV</p>
        </div>
        <span className="small-tag">{badge}</span>
      </div>

      {row ? (
        <>
          <div className="station-detail">
            <span>Recipe type</span>
            <strong>{row.type || '—'}</strong>
          </div>

          <div className="station-detail">
            <span>Total volume</span>
            <strong>{formatAmount(row.total_ml)}</strong>
          </div>

          <div className="station-detail">
            <span>Dispense order</span>
            <strong>{row.dispense_order || '—'}</strong>
          </div>

          <div style={{ marginTop: 18 }}>
            {ingredients.map((item) => (
              <div key={item.name} style={styles.ingredient}>
                <span
                  style={{
                    width: 11,
                    height: 11,
                    flexShrink: 0,
                    borderRadius: 3,
                    background: item.color,
                  }}
                />
                <strong>{item.name}</strong>
                <span
                  style={{
                    marginLeft: 'auto',
                    color: '#f4f8ff',
                  }}
                >
                  {formatAmount(item.amount)}
                </span>
              </div>
            ))}
          </div>
        </>
      ) : (
        <p className="disclaimer">
          Recipe data unavailable.
        </p>
      )}
    </div>
  )
}

function formatAmount(value) {
  return value === undefined || value === null || value === ''
    ? '—'
    : `${value} mL`
}

const styles = {
  page: {
    display: 'flex',
    flexDirection: 'column',
    gap: 20,
    color: '#edf5ff',
    minWidth: 0,
  },

  transitionGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(210px, 1fr))',
    gap: 16,
    alignItems: 'center',
    marginTop: 20,
  },

  previousCard: {
    padding: 20,
    borderRadius: 10,
    border: '1px solid #52677f',
    background: '#102238',
    textAlign: 'center',
    color: '#f4f8ff',
  },

  currentCard: {
    padding: 20,
    borderRadius: 10,
    border: '1px solid #19b9f5',
    background: '#122d46',
    textAlign: 'center',
    color: '#f4f8ff',
  },

  label: {
    display: 'block',
    color: '#9ab9df',
    fontSize: 11,
    letterSpacing: 1.2,
    fontWeight: 600,
  },

  select: {
    display: 'block',
    width: '100%',
    marginTop: 12,
    padding: '11px 12px',
    border: '1px solid #29445e',
    borderRadius: 7,
    background: '#0b192b',
    color: '#f4f8ff',
    colorScheme: 'dark',
    fontSize: 13,
  },

  recipeTitle: {
    marginTop: 16,
    marginBottom: 8,
    color: '#f4f8ff',
    fontSize: 24,
    fontWeight: 700,
  },

  description: {
    color: '#a9c3e3',
    fontSize: 13,
    lineHeight: 1.6,
    marginBottom: 14,
  },

  outgoingBadge: {
    display: 'inline-block',
    padding: '9px 12px',
    border: '1px solid #45647e',
    borderRadius: 6,
    color: '#c5d9ef',
    fontSize: 11,
  },

  incomingText: {
    color: '#f4f8ff',
    fontWeight: 700,
    fontSize: 14,
  },

  arrow: {
    color: '#27baff',
    fontSize: 32,
    textAlign: 'center',
  },

  ingredient: {
    display: 'flex',
    alignItems: 'center',
    gap: 10,
    padding: '12px 10px',
    borderBottom: '1px solid #263f59',
    color: '#f4f8ff',
    fontSize: 13,
  },

  processFlow: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    flexWrap: 'wrap',
    gap: 10,
    padding: '30px 8px',
  },

  processItem: {
    display: 'flex',
    alignItems: 'center',
    gap: 10,
  },

  processBox: {
    padding: 16,
    minWidth: 105,
    textAlign: 'center',
    border: '1px solid #29445e',
    background: '#102238',
    borderRadius: 9,
    color: '#f4f8ff',
  },

  processDescription: {
    color: '#9ab9df',
    fontSize: 12,
    marginTop: 7,
  },

  arrowSmall: {
    color: '#27baff',
    fontSize: 22,
  },

  warningBox: {
    padding: 16,
    border: '1px solid #a96b32',
    background: 'rgba(242, 139, 67, 0.09)',
    borderRadius: 9,
  },

  warningText: {
    color: '#c2d4e9',
    fontSize: 13,
    lineHeight: 1.7,
    marginTop: 7,
  },

  evidenceSource: {
    display: 'flex',
    flexDirection: 'column',
    gap: 5,
    padding: 14,
    background: '#122d46',
    border: '1px solid #31516f',
    borderRadius: 8,
    color: '#eaf3ff',
    fontSize: 12,
    lineHeight: 1.6,
  },

  mismatchNotice: {
    marginTop: 12,
    padding: 15,
    background: 'rgba(242, 139, 67, 0.09)',
    border: '1px solid #a96b32',
    borderRadius: 8,
    color: '#f4f8ff',
    fontSize: 13,
    lineHeight: 1.7,
  },

  evidenceCard: {
    border: '1px solid #29445e',
    background: '#102238',
    borderRadius: 9,
    padding: 16,
    color: '#f4f8ff',
    minWidth: 0,
  },

  evidenceImage: {
    marginTop: 12,
    background: '#071321',
    borderRadius: 7,
    overflow: 'hidden',
  },

  image: {
    width: '100%',
    display: 'block',
    maxHeight: 250,
    objectFit: 'contain',
  },

  alertBox: {
    marginTop: 18,
    padding: 16,
    border: '1px solid #b75b45',
    background: 'rgba(220, 83, 73, 0.10)',
    borderRadius: 9,
  },
}

export default RecipeCarryover
