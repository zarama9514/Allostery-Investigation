# VMD Visualization Instructions for Signal Paths

## Load the colored PDB in VMD:

```tcl
# In VMD Tk Console:

# 1. Load the structure
mol load pdb {mglu3_paths_colored.pdb}

# 2. Remove default representation
mol delrep 0 top

# 3. Add transparent NewCartoon for full protein
mol addrep top
mol modstyle 0 top NewCartoon
mol modcolor 0 top Beta
mol modselect 0 top beta 0
mol modopacity 0 top 0.3

# 4. Add Licorice (sticks) for path residues (beta > 0)
mol addrep top
mol modstyle 1 top Licorice 0.3 30
mol modcolor 1 top Beta
mol modselect 1 top "beta 1.0"

# 5. Add Van der Waals for universal hubs (beta 3.0)
mol addrep top
mol modstyle 2 top VDW
mol modcolor 2 top Beta
mol modselect 2 top "beta 3.0"

# 6. Add Licorice for source/target (beta 5.0)
mol addrep top
mol modstyle 3 top Licorice 0.5 30
mol modcolor 3 top Beta
mol modselect 3 top "beta 5.0"

# 7. Set up color scale (Beta coloring)
# Open: Graphics > Colors > Color Scale
# Set Min/Max: 0.0 / 5.0
# Scheme: BWR (Blue-White-Red) or similar
```

## Color Legend:
- **Beta = 0.0** (transparent gray): Background protein
- **Beta = 1.0** (blue): Only in path A (WITH arrestin)
- **Beta = 2.0** (red): Only in path C (NO arrestin)
- **Beta = 3.0** (white/bright): Universal hubs (299, 300, 509, 532, 537)
- **Beta = 5.0** (black/brightest): Source ligand (901) and target TMD (724)

## Alternative Simple Approach (Python script in VMD):

```python
# Save as vmd_show_paths.py and run: vmd -e vmd_show_paths.py

set pdbfile {mglu3_paths_colored.pdb}
mol load pdb $pdbfile
mol delrep 0 top

# Background
mol addrep top
mol modstyle 0 top NewCartoon
mol modcolor 0 top Beta
mol modselect 0 top "all"

# Color Scale
color Scale method BWR
color Scale min 0.0
color Scale max 5.0

# Adjust representations for visibility
mol modopacity 0 top 0.2
```

## File Location:
`c:\Users\Daniil\IT_projects\Allostery_Investigation\results_4\mfti_2026_ru_v2\12_signal_path\mglu3_paths_colored.pdb`

---

## Quick Start in VMD GUI:
1. File > New Molecule
2. Browse to `mglu3_paths_colored.pdb`
3. Graphics > Representations
4. Delete default rep
5. Add reps as above with Beta coloring
6. Graphics > Color > Color Scale; set to 0.0–5.0
7. Rotate/zoom to see paths highlighted
