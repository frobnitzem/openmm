/* -----------------------------------------------------------------------------
 *              OpenMM(tm) HelloWaterBox example in C++ (June 2009)
 * -----------------------------------------------------------------------------
 * This is a complete, self-contained "hello world" example demonstrating 
 * GPU-accelerated simulation of a system with both bonded and nonbonded forces, 
 * using water (H-O-H) in a periodic box as an example. This is a constant-
 * temperature simulation using an Andersen thermostat. A multi-frame PDB file 
 * is written to stdout which can be read by VMD or other visualization tool to 
 * produce an animation of the resulting trajectory.
 *
 * Pay particular attention to the handling of units in this example. Incorrect
 * handling of units is a very common error; this example shows how you can
 * continue to work with Amber-style units of Angstroms and kCals while correctly
 * communicating with OpenMM in nanometers and kJoules.
 * -------------------------------------------------------------------------- */

#include <cstdio>
#include <string>
#include <vector>
#include <cstdlib>

// -----------------------------------------------------------------------------
//                                 MOCK MD CODE
// -----------------------------------------------------------------------------
// The code starting here and through main() below is meant to represent in 
// simplified form some pre-existing molecular dynamics code, which defines its 
// own data structures for force fields, the atoms in this simulation, and the 
// simulation parameters, and takes care of recording the trajectory. All this 
// has nothing to do with OpenMM; the OpenMM-dependent code comes later and is 
// clearly marked below.
// -----------------------------------------------------------------------------

//                     MODELING AND SIMULATION PARAMETERS
const int    NumWatersAlongEdge  = 10;    // Size of box is NxNxN waters.
const double Temperature         = 300;   // Kelvins
const double FrictionInPerPs     = 91.;   // collisions per picosecond

const bool   UseConstraints      = true;  // Should we constrain O-H bonds?
const double StepSizeInFs        = 1;      // integration step size (fs)
const double ReportIntervalInFs  = 1000;   // how often to generate PDB frame (fs)
const double SimulationTimeInPs  = 10;     // total simulation time (ps)

//                              FORCE FIELD DATA
// For this example we're using a tiny subset of the Amber99 force field.
// We want to keep the data in the original unit system to avoid conversion
// bugs; this requires conversion on the way in and out of OpenMM.

// We only need force field parameters for water here.
const double O_mass             = 15.9994;  // Daltons
const double H_mass             = 1.00794;

//                               PDB FILE WRITER
// This is a PDB writer that only knows how to write out water molecules. It is
// just here for this example and has nothing to do with OpenMM!
static void
myWritePDBFrame(int frameNum, double timeInPs, const std::vector<double>& atomPosInAng) 
{
    const char* atomNames[] = {" O  ", " H1 ", " H2 "}; // cycle through these
    printf("MODEL     %d\n", frameNum);
    printf("REMARK 250 time=%.3f picoseconds\n", timeInPs);
    for (int atom=0; atom < (int)atomPosInAng.size()/3; ++atom) 
    {
        printf("HETATM%5d %4s HOH  %4d    ",        // start of pdb HETATM line
            atom+1, atomNames[atom%3], 1 + atom/3); // atom number, name, residue #
        printf("%8.3f%8.3f%8.3f",                   // middle of pdb HETATM line
            atomPosInAng[3*atom+0], atomPosInAng[3*atom+1], atomPosInAng[3*atom+2]);
        printf("  1.00  0.00            \n");       // end of pdb HETATM line
    }
    printf("ENDMDL\n"); // end of trajectory frame
}


// -----------------------------------------------------------------------------
//                           INTERFACE TO OpenMM
// -----------------------------------------------------------------------------
// These four functions and an opaque structure are used to interface our main
// program with OpenMM without the main program having any direct interaction
// with the OpenMM API. This is a clean approach for interfacing with any MD
// code, although the details of the interface routines will differ. This is
// still just "locally written" code and is not required by OpenMM. Normally 
// these would be in another compilation unit but they are defined later in
// this file.
struct MyOpenMMData;
static MyOpenMMData* myInitializeOpenMM(int    numWatersAlongEdge,
                                        double temperature,
                                        double frictionInPerPs,
                                        double stepSizeInFs, 
                                        std::string& platformName);
static void          myStepWithOpenMM(MyOpenMMData*, int numSteps);
static void          myGetOpenMMState(MyOpenMMData*, double&time, double &en,
                                      std::vector<double>& atomPosInAng);
static void          myTerminateOpenMM(MyOpenMMData*);


// -----------------------------------------------------------------------------
//                           WATER BOX MAIN PROGRAM
// -----------------------------------------------------------------------------
int main() {
    // ALWAYS enclose all OpenMM calls with a try/catch block to make sure that
    // usage and runtime errors are caught and reported.
    try {
        std::string   platformName;

        // Set up OpenMM data structures; return handle and OpenMM Platform name.
        MyOpenMMData* omm = myInitializeOpenMM(NumWatersAlongEdge, Temperature, 
                                               FrictionInPerPs, StepSizeInFs, 
                                               platformName); // output

        // Run the simulation:
        //  (1) Write the first line of the PDB file and the initial configuration.
        //  (2) Run silently entirely within OpenMM between reporting intervals.
        //  (3) Write a PDB frame when the time comes.
        printf("REMARK  Using OpenMM platform %s\n", platformName.c_str());

        std::vector<double> atomPositionsInAng; // x,y,z,x,y,z, ...
        const int NumSilentSteps = (int)(ReportIntervalInFs / StepSizeInFs + 0.5);
        for (int frame=1; ; ++frame) {
            double time, energy;
            myGetOpenMMState(omm, time, energy, atomPositionsInAng);
            printf("%f %f\n", time, energy);
            myWritePDBFrame(frame, time, atomPositionsInAng);

            if (time >= SimulationTimeInPs)
                break;

            myStepWithOpenMM(omm, NumSilentSteps);
        } 
 
        // Clean up OpenMM data structures.
        myTerminateOpenMM(omm);

        return 0; // Normal return from main.
    }

    // Catch and report usage and runtime errors detected by OpenMM and fail.
    catch(const std::exception& e) {
        printf("EXCEPTION: %s\n", e.what());
        return 1;
    }
}


// -----------------------------------------------------------------------------
//                           OpenMM-USING CODE
// -----------------------------------------------------------------------------
// The OpenMM API is visible only at this point and below. Normally this would
// be in a separate compilation module; we're including it here for simplicity.
// -----------------------------------------------------------------------------

// Suppress irrelevant warnings from Microsoft's compiler.
#ifdef _MSC_VER
    #pragma warning(disable:4996)   // sprintf is unsafe 
#endif

#include <OpenMM.h>
#include <OpenMMAmoeba.h>
using OpenMM::Vec3; // so we can just say "Vec3" below

// This is our opaque "handle" class containing all the OpenMM objects that
// must persist from call to call during a simulation. The main program gets 
// a pointer to one of these but sees it as essentially a void* since it 
// doesn't know the definition of this class.
struct MyOpenMMData {
    MyOpenMMData() : system(0), context(0), integrator(0) {}
    ~MyOpenMMData() {delete context; delete integrator; delete system;}
    OpenMM::System*         system;
    OpenMM::Integrator*     integrator;
    OpenMM::Context*        context;
};

void set_positions(OpenMM::Context *ctxt, const double WaterSizeInNm, const int numWatersAlongEdge) {
    std::vector<Vec3> initialPosInNm;

    for (int latticeX = 0; latticeX < numWatersAlongEdge; ++latticeX)
    for (int latticeY = 0; latticeY < numWatersAlongEdge; ++latticeY)
    for (int latticeZ = 0; latticeZ < numWatersAlongEdge; ++latticeZ) {
        // Location of this molecule in the lattice
        Vec3 latticeVec(WaterSizeInNm * latticeX, 
                        WaterSizeInNm * latticeY, 
                        WaterSizeInNm * latticeZ);

        // flip half the waters to prevent giant dipole
        int flip = (rand() % 100) > 49 ? 1 : -1;

        // place this water
        initialPosInNm.push_back(Vec3(0,0,0) + latticeVec);
        initialPosInNm.push_back(Vec3(0.09572*flip,0,0) + latticeVec);
        initialPosInNm.push_back(Vec3(-0.02397*flip,0.09267*flip,0) + latticeVec);
    }
    ctxt->setPositions(initialPosInNm);
}

template <typename T>
std::vector<T> vec3(T a, T b, T c) {
    std::vector<T> x(3);
    x[0] = a;
    x[1] = b;
    x[2] = c;
    return x;
}

template <typename T>
std::vector<T> vec2(T a, T b) {
    std::vector<T> x(2);
    x[0] = a;
    x[1] = b;
    return x;
}

template <typename T>
std::vector<T> vec1(T a) {
    std::vector<T> x(1);
    x[0] = a;
    return x;
}

void addWater(OpenMM::System &system, OpenMM::AmoebaVdwForce &nonbond,
              OpenMM::AmoebaMultipoleForce &es,
              OpenMM::AmoebaBondForce *bondStretch,
              OpenMM::AmoebaAngleForce *bondAngle,
              OpenMM::HarmonicBondForce *bondUB) {
    /* <AmoebaMultipoleForce direct11Scale="0.0" direct12Scale="1.0" direct13Scale="1.0" direct14Scale="1.0" mpole12Scale="0.0" mpole13Scale="0.0" mpole14Scale="0.4" mpole15Scale="0.8" mutual11Scale="1.0" mutual12Scale="1.0" mutual13Scale="1.0" mutual14Scale="1.0" polar12Scale="0.0" polar13Scale="0.0" polar14Intra="0.5" polar14Scale="1.0" polar15Scale="1.0"> */
    const double thole = 0.3900;
    const double damp_pow = 1./6.0;
    double h_dip[] = {-0.00204209484795, 0.0, -0.00307875299958};
    double o_dip[] = {0., 0., 0.00755612136146};
    // q11, q21, q31, q21, q22, q32, q31, q32, q33
    double h_quad[] = {-3.42848248983e-05, 0., -1.89485963908e-06,
                       0., -0.000100240875193, 0.,
                       -1.89485963908e-06, 0., 0.00013452570009};
    double o_quad[] = {0.000354030721139, 0.0, 0.0,
                       0.0, -0.000390257077096, 0.0,
                       0.0, 0., 3.62263559571e-05};
    std::vector<double> o_dipole(o_dip, o_dip+3);
    std::vector<double> o_qpole(o_quad, o_quad+9);
    std::vector<double> h_dipole(h_dip, h_dip+3);
    std::vector<double> h_qpole(h_quad, h_quad+9);

    // Add atom masses to system
    int  oIndex = system.addParticle(O_mass); // O
    int h1Index = system.addParticle(H_mass); // H1
    int h2Index = system.addParticle(H_mass); // H2
    std::vector<int> excl = vec3(oIndex, h1Index, h2Index);

    // Add atom charge, sigma, and stiffness to nonbonded force
    nonbond.addParticle(oIndex, 0.5*0.3405, 0.46024, 1.0); // Oxygen
    //nonbond.addParticle(oIndex, 0.5*0.3405, 0.46024, 0.0); // Oxygen
    nonbond.addParticle(oIndex, 0.5*0.2655, 0.056484, 0.910); // Hydrogen1
    nonbond.addParticle(oIndex, 0.5*0.2655, 0.056484, 0.910); // Hydrogen2

    // Multipoles
    es.addMultipole(-0.51966, o_dipole, o_qpole,
            OpenMM::AmoebaMultipoleForce::Bisector, /* 1 */
            h1Index, h2Index, -1,
            thole, pow(0.000837, damp_pow), 0.000837);
    for(int i=0; i<2; i++) {
        es.addMultipole(0.25983, h_dipole, h_qpole,
                OpenMM::AmoebaMultipoleForce::ZThenX, /* 0 */
                oIndex, h2Index-i, -1,
                thole, pow(0.000496, damp_pow), 0.000496);
    }

    // Covalent Maps
    es.setCovalentMap(oIndex, OpenMM::AmoebaMultipoleForce::Covalent12, // 0
                      vec2(h1Index, h2Index));
    es.setCovalentMap(oIndex,
            OpenMM::AmoebaMultipoleForce::PolarizationCovalent11, excl); // 4
    for(int i=0; i<2; i++) {
        es.setCovalentMap(oIndex, OpenMM::AmoebaMultipoleForce::Covalent12,
                          vec1(oIndex));
        es.setCovalentMap(oIndex, OpenMM::AmoebaMultipoleForce::Covalent13,
                          vec1(h2Index-i));
        es.setCovalentMap(h1Index+i,
                OpenMM::AmoebaMultipoleForce::PolarizationCovalent11, excl); // 4
    }
    
    // Constrain O-H bond lengths or use harmonic forces.
    if (UseConstraints) {
        system.addConstraint(oIndex, h1Index, 0.09572);
        system.addConstraint(oIndex, h2Index, 0.09572);
    } else {
        bondStretch->addBond(oIndex, h1Index, 0.09572, 221584.64);
        bondStretch->addBond(oIndex, h2Index, 0.09572, 221584.64);
    }

    // Note intra-mol. exclusion.
    nonbond.setParticleExclusions(oIndex,  excl);
    nonbond.setParticleExclusions(h1Index, excl);
    nonbond.setParticleExclusions(h2Index, excl);

    // Add bond bend parameters for one angle.
    // See note under bond stretch above regarding the factor of 2 here.
    bondAngle->addAngle(h1Index, oIndex, h2Index, 108.50, 0.0433973816335);
    bondUB->addBond(h1Index, h2Index, 0.15537, 16003.8);
}

// -----------------------------------------------------------------------------
//                      INITIALIZE OpenMM DATA STRUCTURES
// -----------------------------------------------------------------------------
// We take these actions here:
// (1) Load any available OpenMM plugins, e.g. Cuda and Brook.
// (2) Allocate a MyOpenMMData structure to hang on to OpenMM data structures
//     in a manner which is opaque to the caller.
// (3) Fill the OpenMM::System with the force field parameters we want to
//     use and the particular set of atoms to be simulated.
// (4) Create an Integrator and a Context associating the Integrator with
//     the System.
// (5) Select the OpenMM platform to be used.
// (6) Return the MyOpenMMData struct and the name of the Platform in use.
//
// Note that this function must understand the calling MD code's molecule and
// force field data structures so will need to be customized for each MD code.
static MyOpenMMData* 
myInitializeOpenMM( int                 numWatersAlongEdge,
                    double              temperature,
                    double              frictionInPerPs,
                    double              stepSizeInFs, 
                    std::string&        platformName) {
    // Load all available OpenMM plugins from their default location.
    OpenMM::Platform::loadPluginsFromDirectory
       (OpenMM::Platform::getDefaultPluginsDirectory());

    // Allocate space to hold OpenMM objects while we're using them.
    MyOpenMMData* omm = new MyOpenMMData();

    // Create a System and Force objects within the System. Retain a reference
    // to each force object so we can fill in the forces. Note: the System owns
    // the force objects and will take care of deleting them; don't do it yourself!
    OpenMM::System&                 system      = *(omm->system = new OpenMM::System());

    OpenMM::AmoebaVdwForce&         nonbond     = *new OpenMM::AmoebaVdwForce();
    nonbond.setNonbondedMethod(OpenMM::AmoebaVdwForce::CutoffPeriodic);
    // alternative = NoCutoff
    nonbond.setUseDispersionCorrection(true);
    nonbond.setCutoff(0.9);
    nonbond.setSigmaCombiningRule("CUBIC-MEAN");
    nonbond.setEpsilonCombiningRule("HHG");
    // 1-3 scale = 0, 1-4+ scale = 1
    system.addForce(&nonbond);
    
    OpenMM::AmoebaMultipoleForce&   es          = *new OpenMM::AmoebaMultipoleForce();
    es.setNonbondedMethod(OpenMM::AmoebaMultipoleForce::PME);
    es.setPolarizationType(OpenMM::AmoebaMultipoleForce::Mutual);
    // alternative = Direct | Extrapolated
    es.setCutoffDistance(0.9);
    es.setMutualInducedTargetEpsilon(1e-6);
    es.setEwaldErrorTolerance(1e-5);
    system.addForce(&es);

    OpenMM::AmoebaBondForce     *bondStretch;
    if(! UseConstraints) {
        bondStretch = new OpenMM::AmoebaBondForce();
        bondStretch->setAmoebaGlobalBondCubic(-25.5);
        bondStretch->setAmoebaGlobalBondQuartic(379.3125);
        system.addForce(bondStretch);
    }

    OpenMM::AmoebaAngleForce    *bondAngle    = new OpenMM::AmoebaAngleForce();
    bondAngle->setAmoebaGlobalAngleCubic(-0.014);
    bondAngle->setAmoebaGlobalAngleQuartic(5.6e-5);
    bondAngle->setAmoebaGlobalAnglePentic(-7e-7);
    bondAngle->setAmoebaGlobalAngleSextic(2.2e-8);
    system.addForce(bondAngle);

    OpenMM::HarmonicBondForce *bondUB   = new OpenMM::HarmonicBondForce();

    OpenMM::AndersenThermostat   *thermostat  = new OpenMM::AndersenThermostat(
            temperature,      // kelvins
            frictionInPerPs); // collision frequency in 1/picoseconds
    system.addForce(thermostat);
    
    // Volume of one water is 30 Angstroms cubed;
    // Thus length in one dimension is cube-root of 30,
    // or 3.107 Angstroms or 0.3107 nanometers
    const double WaterSizeInNm = 0.3107; // edge of cube containing one water, in nanometers
    // Place water molecules one at a time in an NxNxN rectilinear grid
    const double boxEdgeLengthInNm = WaterSizeInNm * numWatersAlongEdge;

    /*nonbond.setSwitchingDistance(0.8);
    nonbond.setEwaldErrorTolerance(1e-6);*/
    // Create periodic box
    system.setDefaultPeriodicBoxVectors(Vec3(boxEdgeLengthInNm,0,0),
                                  Vec3(0,boxEdgeLengthInNm,0), 
                                  Vec3(0,0,boxEdgeLengthInNm));

    // Specify the atoms and their properties:
    //  (1) System needs to know the masses and constraints (if any).
    //  (2) NonbondedForce needs charges,van der Waals properties (in MD units!).
    
    // Add water molecules one at a time in the NxNxN cubic lattice
    for (int latticeX = 0; latticeX < numWatersAlongEdge; ++latticeX)
    for (int latticeY = 0; latticeY < numWatersAlongEdge; ++latticeY)
    for (int latticeZ = 0; latticeZ < numWatersAlongEdge; ++latticeZ) {
        // Add parameters for one water molecule
        addWater(system, nonbond, es, bondStretch, bondAngle, bondUB);
    }
    
    // Choose an Integrator for advancing time, and a Context connecting the
    // System with the Integrator for simulation. Let the Context choose the
    // best available Platform. Initialize the configuration from the default
    // positions we collected above. Initial velocities will be zero.
    omm->integrator = new OpenMM::VerletIntegrator(StepSizeInFs * OpenMM::PsPerFs);
    //omm->integrator = new OpenMM::LangevinIntegrator(0.0, 0.1, 0.01);
    omm->context    = new OpenMM::Context(*omm->system, *omm->integrator);
    set_positions(omm->context, WaterSizeInNm, numWatersAlongEdge);
    
    platformName = omm->context->getPlatform().getName();
    omm->context->getPlatform().setPropertyValue(*omm->context,
                                "CudaPrecision", "double");
    std::string prec = omm->context->getPlatform().getPropertyValue(
                                *omm->context, "CudaPrecision");
    printf("Precision = %s\n", prec.c_str());
    return omm;
}


// -----------------------------------------------------------------------------
//                     COPY STATE BACK TO CPU FROM OPENMM
// -----------------------------------------------------------------------------
static void
myGetOpenMMState(MyOpenMMData* omm, double& timeInPs, double &energy,
                 std::vector<double>& atomPositionsInAng)
{
    //const OpenMM::State state = omm->context->getState(OpenMM::State::Positions | OpenMM::State::Forces | OpenMM::State::Energy , true);
    const OpenMM::State state = omm->context->getState(OpenMM::State::Positions | OpenMM::State::Energy, true);
    timeInPs = state.getTime(); // OpenMM time is in ps already
    energy   = state.getPotentialEnergy();

    // Copy OpenMM positions into output array and change units from nm to Angstroms.
    const std::vector<Vec3>& positionsInNm = state.getPositions();
    //const std::vector<Vec3>& forces = state.getForces();
    atomPositionsInAng.resize(3*positionsInNm.size());
    for (int i=0; i < (int)positionsInNm.size(); ++i)
        for (int j=0; j < 3; ++j)
            atomPositionsInAng[3*i+j] = positionsInNm[i][j] * OpenMM::AngstromsPerNm;
}


// -----------------------------------------------------------------------------
//                     TAKE MULTIPLE STEPS USING OpenMM 
// -----------------------------------------------------------------------------
static void myStepWithOpenMM(MyOpenMMData* omm, int numSteps) {
    omm->integrator->step(numSteps);
}

// -----------------------------------------------------------------------------
//                     DEALLOCATE OpenMM OBJECTS
// -----------------------------------------------------------------------------
static void 
myTerminateOpenMM(MyOpenMMData* omm) {
    delete omm;
}

