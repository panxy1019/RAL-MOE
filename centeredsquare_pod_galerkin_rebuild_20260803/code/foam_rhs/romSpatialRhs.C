#include "argList.H"
#include "timeSelector.H"
#include "Time.H"
#include "fvMesh.H"
#include "volFields.H"
#include "surfaceFields.H"
#include "fvcDiv.H"
#include "fvcLaplacian.H"
#include "fvcGrad.H"
#include "fvcDdt.H"
#include "fvcFlux.H"

using namespace Foam;

int main(int argc, char *argv[])
{
    timeSelector::addOptions();
    argList::addOption("nu", "scalar", "Kinematic viscosity");
    argList::addBoolOption("noDdt", "Do not evaluate the time derivative");
    #include "setRootCase.H"
    #include "createTime.H"
    instantList timeDirs = timeSelector::select0(runTime, args);
    #include "createMesh.H"

    if (!args.optionFound("nu"))
    {
        FatalErrorInFunction << "Required option -nu is missing" << exit(FatalError);
    }
    const scalar nuValue = args.optionRead<scalar>("nu");
    const dimensionedScalar nu
    (
        "nu",
        dimensionSet(0, 2, -1, 0, 0, 0, 0),
        nuValue
    );

    forAll(timeDirs, timeI)
    {
        runTime.setTime(timeDirs[timeI], timeI);
        mesh.readUpdate();
        Info<< "Time = " << runTime.userTimeName() << nl << endl;

        volVectorField U
        (
            IOobject("U", runTime.name(), mesh, IOobject::MUST_READ),
            mesh
        );
        volScalarField p
        (
            IOobject("p", runTime.name(), mesh, IOobject::MUST_READ),
            mesh
        );
        surfaceScalarField phi
        (
            IOobject("phi", runTime.name(), mesh, IOobject::NO_READ),
            fvc::flux(U)
        );

        volVectorField romConvection
        (
            IOobject("romConvection", runTime.name(), mesh, IOobject::NO_READ, IOobject::AUTO_WRITE),
            fvc::div(phi, U)
        );
        volVectorField romDiffusion
        (
            IOobject("romDiffusion", runTime.name(), mesh, IOobject::NO_READ, IOobject::AUTO_WRITE),
            nu*fvc::laplacian(U)
        );
        volVectorField romPressure
        (
            IOobject("romPressure", runTime.name(), mesh, IOobject::NO_READ, IOobject::AUTO_WRITE),
            fvc::grad(p)
        );
        volVectorField romRhs
        (
            IOobject("romRhs", runTime.name(), mesh, IOobject::NO_READ, IOobject::AUTO_WRITE),
            -romConvection + romDiffusion - romPressure
        );

        romConvection.write();
        romDiffusion.write();
        romPressure.write();
        romRhs.write();
        if (!args.optionFound("noDdt"))
        {
            volVectorField romDdt
            (
                IOobject("romDdt", runTime.name(), mesh, IOobject::NO_READ, IOobject::AUTO_WRITE),
                fvc::ddt(U)
            );
            romDdt.write();
        }
    }
    Info<< "End\n" << endl;
    return 0;
}
