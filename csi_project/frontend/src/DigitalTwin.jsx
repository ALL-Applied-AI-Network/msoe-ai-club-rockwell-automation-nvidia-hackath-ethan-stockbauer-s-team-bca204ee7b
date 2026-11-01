
import { useState } from 'react'

const tanks = [
  {
    id: 'T-100',
    name: 'Clear Diluent',
    color: '#D8E8F1',
    description: 'Clear liquid used to dilute the recipe mixture.',
    role: 'Diluent',
    status: 'ACTIVE',
  },
  {
    id: 'T-200',
    name: 'Orange Reagent',
    color: '#F08A43',
    description: 'Orange reagent used in recipes requiring T-200.',
    role: 'Color reagent',
    status: 'ACTIVE',
  },
  {
    id: 'T-300',
    name: 'Unavailable Tank',
    color: '#64748B',
    description: 'Out of service during dataset collection. Do not use.',
    role: 'Offline',
    status: 'OFFLINE',
  },
  {
    id: 'T-400',
    name: 'Blue Reagent',
    color: '#398BDB',
    description: 'Blue reagent used in recipes requiring T-400.',
    role: 'Color reagent',
    status: 'ACTIVE',
  },
]

// Illustrative recipe references from the existing dashboard component.
// These values are not dynamically read from the recipe CSV.
const recipes = {
  '5': {
    name: 'Recipe 5',
    order: 'T-200 → T-400 → T-100',
    volume: '30 mL',
    tanks: ['T-100', 'T-200', 'T-400'],
  },
  '6': {
    name: 'Recipe 6',
    order: 'T-200 → T-100',
    volume: '40 mL',
    tanks: ['T-100', 'T-200'],
  },
  '7': {
    name: 'Recipe 7',
    order: 'T-400 → T-100',
    volume: '70 mL',
    tanks: ['T-100', 'T-400'],
  },
}

export default function DigitalTwin() {
  const [selectedTank, setSelectedTank] = useState('T-200')
  const [selectedRecipe, setSelectedRecipe] = useState('6')

  const recipe = recipes[selectedRecipe]
  const tank = tanks.find((item) => item.id === selectedTank)

  const activeTanks = recipe.tanks

  // Build the mixing-path color from the ingredients in the selected recipe.
  const recipeColors = recipe.tanks
    .map((id) => tanks.find((item) => item.id === id)?.color)
    .filter(Boolean)

  const mixingGradient =
    recipeColors.length > 1
      ? `linear-gradient(90deg, ${recipeColors.join(', ')})`
      : recipeColors[0] || '#64748B'

  const selectedTankIsInRecipe = activeTanks.includes(selectedTank)

  return (
    <div style={styles.page}>

      {/* STATION PROCESS VISUALIZATION */}
      <section className="panel">
        <div className="panel-heading">
          <div>
            <h3>Station Process Visualization</h3>
            <p>Tank supply → shared mixing path → vial filling</p>
          </div>
          <span className="online-tag">SCHEMATIC VIEW</span>
        </div>

        <div style={styles.diagram}>

          {/* TANKS */}
          <div style={styles.tankGrid}>
            {tanks.map((item) => {
              const active = activeTanks.includes(item.id)
              const selected = selectedTank === item.id
              const offline = item.status === 'OFFLINE'

              return (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => setSelectedTank(item.id)}
                  aria-pressed={selected}
                  style={{
                    ...styles.tankButton,
                    background: selected ? '#1A3855' : '#102238',
                    border: selected
                      ? '2px solid #39BDF8'
                      : active
                        ? '1px solid #3189B8'
                        : '1px solid #29435D',
                    opacity: offline ? 0.55 : 1,
                    boxShadow: selected
                      ? '0 0 0 2px rgba(57,189,248,.12)'
                      : 'none',
                  }}
                >
                  <div style={styles.tankVisual}>
                    <div
                      style={{
                        ...styles.tankLiquid,
                        height: offline ? '0%' : active ? '65%' : '25%',
                        background: item.color,
                        opacity: active ? 1 : 0.35,
                      }}
                    />
                  </div>

                  <strong>{item.id}</strong>

                  <div style={styles.tankRole}>
                    {item.role}
                  </div>

                  <div
                    style={{
                      ...styles.tankStatus,
                      color: offline
                        ? '#8499AD'
                        : active
                          ? '#42C7FF'
                          : '#8499AD',
                    }}
                  >
                    {offline
                      ? 'OFFLINE'
                      : active
                        ? 'IN RECIPE'
                        : 'NOT SELECTED'}
                  </div>

                  {selected && (
                    <span style={styles.selectedLabel}>
                      SELECTED
                    </span>
                  )}
                </button>
              )
            })}
          </div>

          <div style={styles.flowArrow}>↓ ↓ ↓</div>

          {/* MIXING CHAMBER */}
          <div style={styles.mixingChamber}>
            <div style={styles.cyanLabel}>SHARED FLUID PATH</div>

            <strong style={styles.chamberTitle}>
              Mixing Chamber
            </strong>

            <p style={styles.chamberDescription}>
              Recipe {selectedRecipe} uses {activeTanks.length} active
              ingredient tanks.
            </p>

            <div style={styles.colorFlow}>
              <div
                style={{
                  ...styles.colorFlowFill,
                  background: mixingGradient,
                }}
              />
            </div>

            <div style={styles.ingredientLegend}>
              {recipe.tanks.map((id) => {
                const item = tanks.find((entry) => entry.id === id)

                return (
                  <span key={id} style={styles.legendItem}>
                    <i
                      style={{
                        ...styles.legendDot,
                        background: item.color,
                      }}
                    />
                    {id}
                  </span>
                )
              })}
            </div>

            <div style={styles.residualNotice}>
              POSSIBLE RESIDUAL FLUID AFTER RECIPE CHANGE
            </div>
          </div>

          <div style={styles.flowArrow}>↓</div>

          {/* FILLING NOZZLE */}
          <div style={styles.nozzle}>
            <strong style={styles.nozzleTitle}>Filling Nozzle</strong>

            <div style={styles.nozzleStem} />

            <div style={styles.vial}>
              <div
                style={{
                  ...styles.vialLiquid,
                  background: mixingGradient,
                }}
              />
            </div>

            <div style={styles.nozzleCaption}>
              VIAL FILL STATION
            </div>

            <span style={styles.nozzleRecipe}>
              RECIPE {selectedRecipe}
            </span>
          </div>
        </div>

        <p className="disclaimer">
          Conceptual process schematic based on Dataset 2.
          Tank levels and fluid movement are illustrative, not live sensor readings.
        </p>
      </section>

      {/* TANK INFORMATION AND RECIPE FLOW */}
      <section className="content-grid">

        {/* SELECTED TANK DETAILS */}
        <div className="panel">
          <div className="panel-heading">
            <div>
              <h3>Tank Information</h3>
              <p>Click a tank in the diagram to inspect it.</p>
            </div>
            <span className="small-tag">{selectedTank}</span>
          </div>

          <div style={styles.tankInfo}>
            <div style={styles.tankInfoHeader}>
              <div
                style={{
                  ...styles.tankInfoColor,
                  background: tank.color,
                }}
              />
              <strong style={styles.whiteText}>{tank.id}</strong>
              <span
                style={{
                  ...styles.statusBadge,
                  color: tank.status === 'OFFLINE' ? '#ff9ca9' : '#78e6b5',
                  borderColor:
                    tank.status === 'OFFLINE' ? '#854452' : '#287a60',
                }}
              >
                {tank.status}
              </span>
            </div>

            <h3 style={styles.tankInfoTitle}>{tank.name}</h3>

            <p style={styles.tankDescription}>
              {tank.description}
            </p>

            <div className="station-detail">
              <span>Tank role</span>
              <strong>{tank.role}</strong>
            </div>

            <div className="station-detail">
              <span>Status</span>
              <strong className={tank.status === 'OFFLINE' ? 'red' : 'green'}>
                {tank.status}
              </strong>
            </div>

            <div className="station-detail">
              <span>Current recipe</span>
              <strong>
                {selectedTankIsInRecipe
                  ? `Recipe ${selectedRecipe}`
                  : 'Not included'}
              </strong>
            </div>

            {!selectedTankIsInRecipe && tank.status !== 'OFFLINE' && (
              <div style={styles.inactiveNotice}>
                This tank is not used in the currently selected recipe.
              </div>
            )}

            {tank.status === 'OFFLINE' && (
              <div style={styles.offlineNotice}>
                This tank was unavailable during dataset collection.
                It is not included in the recipe flow.
              </div>
            )}
          </div>
        </div>

        {/* RECIPE FLOW */}
        <div className="panel">
          <div className="panel-heading">
            <div>
              <h3>Recipe Flow</h3>
              <p>Explore how different recipes use the tanks.</p>
            </div>
            <span className="small-tag">ILLUSTRATIVE DATA</span>
          </div>

          <label htmlFor="digital-twin-recipe" style={styles.recipeLabel}>
            SELECT RECIPE
          </label>

          <select
            id="digital-twin-recipe"
            value={selectedRecipe}
            onChange={(event) => setSelectedRecipe(event.target.value)}
            style={styles.recipeSelect}
          >
            <option value="5">Recipe 5</option>
            <option value="6">Recipe 6</option>
            <option value="7">Recipe 7</option>
          </select>

          <h3 style={styles.recipeTitle}>{recipe.name}</h3>

          <div className="station-detail">
            <span>Total volume</span>
            <strong>{recipe.volume}</strong>
          </div>

          <div className="station-detail">
            <span>Dispense order</span>
            <strong>{recipe.order}</strong>
          </div>

          <div style={styles.recipeTankList}>
            {recipe.tanks.map((id) => {
              const item = tanks.find((entry) => entry.id === id)

              return (
                <span
                  key={id}
                  style={styles.recipeTankTag}
                >
                  <i
                    style={{
                      ...styles.legendDot,
                      background: item.color,
                    }}
                  />
                  {id}
                </span>
              )
            })}
          </div>

          <div style={styles.recipeNote}>
            Changing the recipe updates the illustrative tank selection
            and mixing-path colors.
          </div>

          <p className="disclaimer">
            Recipe values shown here are hardcoded illustrative references
            from the existing component. They are not dynamically loaded
            from the recipe CSV.
          </p>
        </div>
      </section>

      {/* CARRYOVER AWARENESS */}
      <section className="panel">
        <div className="panel-heading">
          <div>
            <h3>Carryover Awareness</h3>
            <p>Why the shared mixing path matters.</p>
          </div>
          <span className="alert">CHANGEOVER RISK</span>
        </div>

        <div className="alert-row">
          <span className="alert-icon">!</span>
          <div>
            <strong>
              Previous recipe residue may affect the next fill.
            </strong>
            <p>
              Fluid remaining in the shared mixing path can change the
              appearance of the first vial after a recipe change.
            </p>
          </div>
        </div>

        <p className="disclaimer">
          This is an operational explanation, not a live prediction of
          residual fluid quantity or a confirmed defect.
        </p>
      </section>
    </div>
  )
}

const styles = {
  page: {
    display: 'grid',
    gap: 20,
    minWidth: 0,
  },

  diagram: {
    background: '#0A1728',
    border: '1px solid #203B55',
    borderRadius: 12,
    padding: 20,
    overflowX: 'auto',
  },

  tankGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(4, minmax(110px, 1fr))',
    gap: 12,
  },

  tankButton: {
    position: 'relative',
    borderRadius: 10,
    padding: 13,
    color: '#FFFFFF',
    cursor: 'pointer',
    minWidth: 0,
    textAlign: 'center',
    transition: 'background .2s ease, border .2s ease',
  },

  tankVisual: {
    height: 65,
    width: 43,
    margin: '0 auto 10px',
    border: '2px solid #A8BED1',
    borderRadius: '8px 8px 13px 13px',
    overflow: 'hidden',
    display: 'flex',
    alignItems: 'end',
  },

  tankLiquid: {
    width: '100%',
    transition: 'height .25s ease, opacity .25s ease',
  },

  tankRole: {
    color: '#AFC3D6',
    fontSize: 11,
    marginTop: 5,
  },

  tankStatus: {
    fontSize: 10,
    fontWeight: 700,
    marginTop: 8,
  },

  selectedLabel: {
    display: 'block',
    marginTop: 7,
    color: '#55caff',
    fontSize: 9,
    letterSpacing: 0.7,
  },

  flowArrow: {
    textAlign: 'center',
    color: '#42BDF5',
    fontSize: 22,
    margin: '12px 0',
  },

  mixingChamber: {
    maxWidth: 500,
    margin: '0 auto',
    padding: 20,
    border: '1px solid #3189B8',
    borderRadius: 10,
    background: '#102A43',
    textAlign: 'center',
  },

  cyanLabel: {
    color: '#42BDF5',
    fontSize: 11,
    letterSpacing: 1,
    fontWeight: 700,
  },

  chamberTitle: {
    display: 'block',
    color: '#FFFFFF',
    fontSize: 18,
    marginTop: 7,
  },

  chamberDescription: {
    color: '#AFC3D6',
    fontSize: 12,
    lineHeight: 1.6,
  },

  colorFlow: {
    height: 12,
    borderRadius: 10,
    overflow: 'hidden',
    background: '#172a40',
    marginTop: 16,
    border: '1px solid #48627a',
  },

  colorFlowFill: {
    width: '100%',
    height: '100%',
    transition: 'background .25s ease',
  },

  ingredientLegend: {
    display: 'flex',
    justifyContent: 'center',
    flexWrap: 'wrap',
    gap: 15,
    marginTop: 13,
  },

  legendItem: {
    display: 'flex',
    alignItems: 'center',
    gap: 6,
    color: '#d7e6f6',
    fontSize: 11,
  },

  legendDot: {
    display: 'inline-block',
    width: 9,
    height: 9,
    borderRadius: 3,
  },

  residualNotice: {
    color: '#F3B45D',
    fontSize: 11,
    fontWeight: 700,
    marginTop: 15,
    lineHeight: 1.5,
  },

  nozzle: {
    maxWidth: 330,
    margin: '0 auto',
    padding: 17,
    background: '#102238',
    border: '1px solid #29435D',
    borderRadius: 10,
    textAlign: 'center',
  },

  nozzleTitle: {
    color: '#FFFFFF',
  },

  nozzleStem: {
    width: 5,
    height: 30,
    background: '#A8BED1',
    margin: '8px auto',
  },

  vial: {
    position: 'relative',
    width: 50,
    height: 65,
    border: '2px solid #A8BED1',
    borderRadius: '5px 5px 12px 12px',
    margin: '0 auto',
    overflow: 'hidden',
    background: 'rgba(255,255,255,.04)',
  },

  vialLiquid: {
    position: 'absolute',
    left: 0,
    right: 0,
    bottom: 0,
    height: '45%',
    transition: 'background .25s ease',
  },

  nozzleCaption: {
    color: '#AFC3D6',
    fontSize: 11,
    marginTop: 9,
  },

  nozzleRecipe: {
    display: 'inline-block',
    color: '#55caff',
    fontSize: 10,
    fontWeight: 700,
    marginTop: 7,
  },

  tankInfo: {
    padding: 16,
    background: '#102238',
    borderRadius: 10,
    border: '1px solid #29435D',
  },

  tankInfoHeader: {
    display: 'flex',
    alignItems: 'center',
    gap: 12,
  },

  tankInfoColor: {
    width: 20,
    height: 20,
    borderRadius: 5,
  },

  whiteText: {
    color: '#FFFFFF',
  },

  statusBadge: {
    marginLeft: 'auto',
    padding: '5px 8px',
    border: '1px solid',
    borderRadius: 5,
    fontSize: 10,
    fontWeight: 700,
  },

  tankInfoTitle: {
    color: '#FFFFFF',
    marginTop: 15,
    fontSize: 18,
  },

  tankDescription: {
    color: '#AFC3D6',
    fontSize: 13,
    lineHeight: 1.7,
  },

  inactiveNotice: {
    padding: 11,
    marginTop: 14,
    background: 'rgba(72,186,255,.08)',
    border: '1px solid #315b79',
    borderRadius: 7,
    color: '#b9d8f1',
    fontSize: 11,
    lineHeight: 1.6,
  },

  offlineNotice: {
    padding: 11,
    marginTop: 14,
    background: 'rgba(239,68,68,.08)',
    border: '1px solid #854452',
    borderRadius: 7,
    color: '#ffadb5',
    fontSize: 11,
    lineHeight: 1.6,
  },

  recipeLabel: {
    display: 'block',
    color: '#AFC3D6',
    fontSize: 12,
    marginBottom: 8,
  },

  recipeSelect: {
    width: '100%',
    padding: 12,
    background: '#102238',
    color: '#FFFFFF',
    colorScheme: 'dark',
    border: '1px solid #29435D',
    borderRadius: 8,
  },

  recipeTitle: {
    color: '#FFFFFF',
    marginTop: 18,
  },

  recipeTankList: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: 8,
    marginTop: 16,
  },

  recipeTankTag: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: 7,
    padding: '7px 10px',
    border: '1px solid #31516f',
    borderRadius: 6,
    background: '#172a40',
    color: '#eaf3ff',
    fontSize: 11,
  },

  recipeNote: {
    marginTop: 15,
    padding: 12,
    background: 'rgba(32,184,213,.08)',
    border: '1px solid #246078',
    borderRadius: 7,
    color: '#b9d8f1',
    fontSize: 11,
    lineHeight: 1.6,
  },
}
