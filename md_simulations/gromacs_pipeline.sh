#!/bin/bash

PROTEIN=$1

# --- Configuration ---
PDB_INPUT="$PROTEIN.pdb"
FF="amber99sb-ildn"
WATER="tip3p"
OUT_DIR="data"
MDP_DIR="./mdp"
BOX=dodecahedron   # changes amount of water, see description for other options
BOX_DIST=1.0       # box edge distance in nm
CONC=0.1           # molar NaCl concentration

# Create output directory if it doesn't exist
mkdir -p $OUT_DIR

# 1. Structure Conversion
echo "******************************************"
echo "Step 1: Generating Topology..."
gmx pdb2gmx -f $PDB_INPUT -o $OUT_DIR/processed.gro -p $OUT_DIR/topol.top -ff $FF -water $WATER
mv posre.itp data/

# 2. Creating the Box & Solvating
echo "******************************************"
echo "Step 2: Defining Box and Adding Solvent..."
gmx editconf -f $OUT_DIR/processed.gro -o $OUT_DIR/boxed.gro -bt $BOX -d $BOX_DIST
gmx solvate -cp $OUT_DIR/boxed.gro -cs spc216.gro -o $OUT_DIR/solvated.gro -p $OUT_DIR/topol.top

# 3. Adding Ions
echo "******************************************"
echo "Step 3: Adding Ions..."
gmx grompp -f $MDP_DIR/ions.mdp -c $OUT_DIR/solvated.gro -p $OUT_DIR/topol.top -o $OUT_DIR/ions.tpr -po $OUT_DIR/ions_out.mdp
# Replaces SOL with ions to neutralize
echo "SOL" | gmx genion -s $OUT_DIR/ions.tpr -o $OUT_DIR/ionized.gro -p $OUT_DIR/topol.top -pname NA -nname CL -neutral -conc $CONC

# 4. Energy Minimization
echo "******************************************"
echo "Step 4: Energy Minimization..."
gmx grompp -f $MDP_DIR/minim.mdp -c $OUT_DIR/ionized.gro -p $OUT_DIR/topol.top -o $OUT_DIR/em.tpr -po $OUT_DIR/em_out.mdp
gmx mdrun -v -deffnm $OUT_DIR/em
grep "Steepest Descents converged to Fmax" $OUT_DIR/em.log

# 5. NVT Equilibration
echo "******************************************"
echo "Step 5: NVT Equilibration..."
gmx grompp -f $MDP_DIR/nvt.mdp -c $OUT_DIR/em.gro -r $OUT_DIR/em.gro -p $OUT_DIR/topol.top -o $OUT_DIR/nvt.tpr -po $OUT_DIR/nvt_out.mdp
gmx mdrun -v -deffnm $OUT_DIR/nvt

# 6. NPT Equilibration
echo "******************************************"
echo "Step 6: NPT Equilibration..."
gmx grompp -f $MDP_DIR/npt.mdp -c $OUT_DIR/nvt.gro -r $OUT_DIR/nvt.gro -t $OUT_DIR/nvt.cpt -p $OUT_DIR/topol.top -o $OUT_DIR/npt.tpr -po $OUT_DIR/npt_out.mdp
gmx mdrun -v -deffnm $OUT_DIR/npt

# 7. Production MD
echo "******************************************"
echo "Step 7: Production Run..."
gmx grompp -f $MDP_DIR/md.mdp -c $OUT_DIR/npt.gro -t $OUT_DIR/npt.cpt -p $OUT_DIR/topol.top -o $OUT_DIR/production.tpr -po $OUT_DIR/md_out.mdp
gmx mdrun -v -deffnm $OUT_DIR/production
