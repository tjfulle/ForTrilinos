/*
 * Copyright 2017-2018, UT-Battelle, LLC
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * License-Filename: LICENSE
 */
#include <Teuchos_Comm.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_FancyOStream.hpp>
#include <Teuchos_AbstractFactoryStd.hpp>

#include <Tpetra_Core.hpp>
#include <Tpetra_Vector.hpp>

#include "Tpetra_ModelEvaluator_1DFEM.hpp"
#include "nox_solver.hpp"

template <class Scalar>
void fill_analytic_solution(Tpetra::MultiVector<Scalar>& x)
{
  // Analytic solution from Mathematica:
  /*
    sol=NDSolve[{x''[t]==x[t]*x[t],x[0]=1,x'[1]=0}, x[t], {t,0,1}];
  */
  Scalar global_solution[101] =
    {1.,0.993525,0.987148,0.980869,0.974686,0.968598,0.962604,0.956702,0.950892,
     0.945173,0.939542,0.934001,0.928546,0.923177,0.917894,0.912695,0.907579,0.902546,
     0.897594,0.892723,0.887931,0.883218,0.878584,0.874026,0.869545,0.865139,0.860809,
     0.856552,0.852369,0.848258,0.84422,0.840252,0.836355,0.832529,0.828771,0.825082,
     0.821462,0.817908,0.814422,0.811002,0.807648,0.804359,0.801134,0.797974,0.794878,
     0.791844,0.788874,0.785966,0.783119,0.780334,0.777609,0.774946,0.772342,0.769798,
     0.767313,0.764887,0.762519,0.76021,0.757958,0.755764,0.753627,0.751547,0.749523,
     0.747555,0.745644,0.743788,0.741987,0.740241,0.73855,0.736914,0.735332,0.733804,
     0.73233,0.730909,0.729542,0.728228,0.726968,0.72576,0.724604,0.723501,0.722451,
     0.721453,0.720507,0.719612,0.71877,0.717979,0.71724,0.716552,0.715915,0.71533,
     0.714796,0.714313,0.713881,0.7135,0.71317,0.712891,0.712662,0.712485,0.712358,
     0.712282,0.712256
    };
  using GO = typename Tpetra::MultiVector<Scalar>::global_ordinal_type;
  auto Invalid = Teuchos::OrdinalTraits<GO>::invalid();
  for (int i = 0; i < 101; i++)
  {
    auto row = x.getMap()->getGlobalElement(i);
    if (row != Invalid)
      x.replaceGlobalValue(row, 0, global_solution[i]);
  }
}

// Sets up and runs the nonlinear optimization in NOX
int main2(Teuchos::RCP<const Teuchos::Comm<int>>& comm,
          Teuchos::RCP<Teuchos::ParameterList>& plist)
{

  using Teuchos::RCP;
  using Teuchos::rcp;

  // Get default Tpetra template types
  using Scalar = Tpetra::CrsMatrix<>::scalar_type;
  using LO = Tpetra::CrsMatrix<>::local_ordinal_type;
  using GO = Tpetra::CrsMatrix<>::global_ordinal_type;
  using Node = Tpetra::CrsMatrix<>::node_type;

  Teuchos::TimeMonitor::zeroOutTimers();

  // Create the model evaluator object
  RCP<TpetraModelEvaluator1DFEM<Scalar,LO,GO,Node>> evaluator =
    rcp(new TpetraModelEvaluator1DFEM<Scalar,LO,GO,Node>(comm, 100, 0.0, 1.0));
  evaluator->setup(plist);

  ForTrilinos::NOXSolver<Scalar,LO,GO,Node> nox_solver(evaluator);
  nox_solver.setup(plist);
  NOX::StatusTest::StatusType solve_status = nox_solver.solve();

  int returncode = (solve_status == NOX::StatusTest::Converged) ? 0 : 1;

  auto solution = nox_solver.get_solution();
  Tpetra::MultiVector<Scalar> analytic_solution(solution->getMap(), 1);
  fill_analytic_solution(analytic_solution);

  Teuchos::TimeMonitor::summarize();

  return returncode;
}

int main(int argc, char* argv[])
{

  // GlobalMPISession calls MPI_Init() in its constructor, if
  // appropriate.
  Teuchos::GlobalMPISession mpiSession(&argc, &argv, NULL);

  // Get a communicator corresponding to MPI_COMM_WORLD
  Teuchos::RCP<const Teuchos::Comm<int>> comm = Tpetra::getDefaultComm();

  Teuchos::RCP<Teuchos::ParameterList> params = Teuchos::createParameterList("Test Params");
  Teuchos::updateParametersFromXmlFileAndBroadcast("nox_params.xml", params.ptr(), *comm);

  int errors = main2(comm, params);

  TEUCHOS_TEST_FOR_EXCEPTION(errors!=0,
   std::runtime_error,
   "One or more NOX evaluations failed!");

  if (comm->getRank() == 0) {
    std::cout << "End Result: TEST PASSED" << std::endl;
  }
  return 0;
}
