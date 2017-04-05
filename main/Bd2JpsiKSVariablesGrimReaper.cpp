/******************************************/
// Bd2JpsiKSVariablesGrimReaper.cpp
//
// Standalone GrimReaper that adds a lot of
// common variables to trees.
//
// Author: Christopher Hasenberg
// Date: 2017-04-4
/******************************************/

// from STL
#include <tuple>
#include <list>

// from ROOT
#include "TRandom3.h"
#include "TCut.h"

// from DooCore
#include "doocore/io/MsgStream.h"
#include "doocore/config/Summary.h"

// from DooSelection
#include "dooselection/reducer/Reducer.h"
#include "dooselection/reducer/ReducerLeaf.h"
#include "dooselection/reducer/KinematicReducerLeaf.h"

using namespace dooselection::reducer;
using namespace doocore::io;

// forward declarations
// typedef tuple: head, daughters, stable particles, isMC, isFlat
typedef std::tuple<std::string, std::list<std::string>, std::list<std::string>, bool, bool> cfg_tuple;
cfg_tuple Configure(Reducer* _rdcr, std::string& _channel);
void MCLeaves(Reducer* _rdcr, cfg_tuple& cfg);
void MassLeaves(Reducer* _rdcr, cfg_tuple& cfg);
void TimeLeaves(Reducer* _rdcr, cfg_tuple& cfg);
void TriggerLeaves(Reducer* _rdcr, cfg_tuple& cfg);
void VetoLeaves(Reducer* _rdcr, cfg_tuple& cfg);
void AuxiliaryLeaves(Reducer* _rdcr, cfg_tuple& cfg);

int main(int argc, char * argv[]){
  sinfo << "-info-  \t" << "Bd2JpsiKSVariablesGrimReaper \t" << "Welcome!" << endmsg;
  std::string inputfile, inputtree, outputfile, outputtree, decay_channel;
  if (argc == 5){
    inputfile = argv[1];
    inputtree = argv[2];
    outputfile = argv[3];
    outputtree = argv[4];
    decay_channel = "Bd2JpsiKS";
  }
  else{
    serr << "-ERROR- \t" << "Bd2JpsiKSVariablesGrimReaper \t" << "Parameters needed:" << endmsg;
    serr << "-ERROR- \t" << "Bd2JpsiKSVariablesGrimReaper \t"<< "input_file_name input_tree_name output_file_name output_tree_name" << endmsg;
    return 1;
  }

  Reducer* reducer = new Reducer();
  doocore::config::Summary& summary = doocore::config::Summary::GetInstance();
  summary.AddSection("I/O");
  summary.Add("Input file", inputfile);
  summary.Add("Input tree", inputtree);
  summary.Add("Output file", outputfile);
  summary.Add("Output file", outputtree);

  // reducer part
  reducer->set_input_file_path(inputfile);
  reducer->set_input_tree_path(inputtree);
  reducer->set_output_file_path(outputfile);
  reducer->set_output_tree_path(outputtree);

  reducer->Initialize();

  // config
  cfg_tuple cfg = Configure(reducer, decay_channel);

  // add leaves
  summary.AddSection("Added leaves");
  VetoLeaves(reducer, cfg);

  reducer->Run();
  reducer->Finalize();
}

cfg_tuple Configure(Reducer* _rdcr, std::string& _channel){
  doocore::config::Summary& summary = doocore::config::Summary::GetInstance();
  summary.AddSection("Channel");
  // typedef tuple: head, daughters, stable particles, isMC, isFlat
  std::string head ="";
  std::list<std::string> daughters;
  std::list<std::string> stable_particles;
  bool isMC = false;
  bool isFlat = false;
  if (_channel == "Bd2JpsiKS"){
    head = "B0";
    daughters.push_back("J_psi_1S");
    daughters.push_back("KS0");
    stable_particles.push_back("muminus");
    stable_particles.push_back("muplus");
    stable_particles.push_back("piminus");
    stable_particles.push_back("pplus");
    isMC = _rdcr->LeafExists(head+"_BKGCAT");
    isFlat = (_rdcr->LeafExists("flat_array_index") || _rdcr->LeafExists("idxPV"));
  }
  else{
    serr << "-ERROR- \t" << "Bd2JpsiKSVariablesGrimReaper \t" << "No valid decay channel. Possible decay channels are:" << endmsg;
    serr << "-ERROR- \t" << "Bd2JpsiKSVariablesGrimReaper \t" << "- Bd2JspiKS" << endmsg;
  }
  summary.Add("Name", _channel);
  summary.Add("Head", head);
  for (std::list<std::string>::iterator it = daughters.begin(); it != daughters.end(); ++it){
    summary.Add("Daughter", *it);
  }
  for (std::list<std::string>::iterator it = stable_particles.begin(); it != stable_particles.end(); ++it){
    summary.Add("Stable", *it);
  }
  summary.AddSection("Data Type");
  summary.Add("MC", isMC);
  summary.Add("Flat", isFlat);

  if (isFlat) sinfo << "-info-  \t" << "You are running the reducer over a flat tuple!" << endmsg;
  if (isMC) sinfo << "-info-  \t" << "You are running the reducer over a MC tuple!" << endmsg;

  return std::make_tuple(head, daughters, stable_particles, isMC, isFlat);
}


void VetoLeaves(Reducer* _rdcr, cfg_tuple& cfg){
  doocore::config::Summary& summary = doocore::config::Summary::GetInstance();
  // handle flattened tuples
  std::string flat_suffix = "";
  if (std::get<4>(cfg)) flat_suffix = "_flat";

  // veto leafs
  std::string pplus_px, pplus_py, pplus_pz;
  std::string piminus_px, piminus_py, piminus_pz;
  std::string piplus_px, piplus_py, piplus_pz;
  std::string pminus_px, pminus_py, pminus_pz;


  std::string mass_hypo_constraints = "";
  if (_rdcr->LeafExists(std::get<0>(cfg)+"_FitDaughtersConst_KS0_P0_PX")){
    pplus_px  = std::get<0>(cfg)+"_FitDaughtersConst_KS0_P0_PX"+flat_suffix;
    pplus_py  = std::get<0>(cfg)+"_FitDaughtersConst_KS0_P0_PY"+flat_suffix;
    pplus_pz  = std::get<0>(cfg)+"_FitDaughtersConst_KS0_P0_PZ"+flat_suffix;
    piminus_px = std::get<0>(cfg)+"_FitDaughtersConst_KS0_P1_PX"+flat_suffix;
    piminus_py = std::get<0>(cfg)+"_FitDaughtersConst_KS0_P1_PY"+flat_suffix;
    piminus_pz = std::get<0>(cfg)+"_FitDaughtersConst_KS0_P1_PZ"+flat_suffix;
    piplus_px  = std::get<0>(cfg)+"_FitDaughtersConst_KS0_P0_PX"+flat_suffix;
    piplus_py  = std::get<0>(cfg)+"_FitDaughtersConst_KS0_P0_PY"+flat_suffix;
    piplus_pz  = std::get<0>(cfg)+"_FitDaughtersConst_KS0_P0_PZ"+flat_suffix;
    pminus_px = std::get<0>(cfg)+"_FitDaughtersConst_KS0_P1_PX"+flat_suffix;
    pminus_py = std::get<0>(cfg)+"_FitDaughtersConst_KS0_P1_PY"+flat_suffix;
    pminus_pz = std::get<0>(cfg)+"_FitDaughtersConst_KS0_P1_PZ"+flat_suffix;
    mass_hypo_constraints = "JpsiPV";
  }
  else if (_rdcr->LeafExists("pplus_PX")){
    pplus_px  = "pplus_PX";
    pplus_py  = "pplus_PY";
    pplus_pz  = "pplus_PZ";
    piminus_px = "piminus_PX";
    piminus_py = "piminus_PY";
    piminus_pz = "piminus_PZ";
    piplus_px  = "pplus_PX";
    piplus_py  = "pplus_PY";
    piplus_pz  = "pplus_PZ";
    pminus_px = "piminus_PX";
    pminus_py = "piminus_PY";
    pminus_pz = "piminus_PZ";
    mass_hypo_constraints = "NoConstr";
  }

if (mass_hypo_constraints!=""){
    // mass hypotheses// mass hypotheses
    KinematicReducerLeaf<Double_t>* varLambda0MassHypo_ppluspiminus = new KinematicReducerLeaf<Double_t>("varLambda0MassHypo_ppluspiminus", "varLambda0MassHypo_ppluspiminus", "Double_t", NULL);
    varLambda0MassHypo_ppluspiminus->FixedMassDaughtersTwoBodyDecayMotherMass(
        _rdcr->GetInterimLeafByName(pplus_px),
        _rdcr->GetInterimLeafByName(pplus_py),
        _rdcr->GetInterimLeafByName(pplus_pz),
        938.272046,
        _rdcr->GetInterimLeafByName(piminus_px),
        _rdcr->GetInterimLeafByName(piminus_py),
        _rdcr->GetInterimLeafByName(piminus_pz),
        139.57018);
    _rdcr->RegisterDoubleLeaf(varLambda0MassHypo_ppluspiminus);

    KinematicReducerLeaf<Double_t>* varLambda0MassHypo_pminuspiplus = new KinematicReducerLeaf<Double_t>("varLambda0MassHypo_pminuspiplus", "varLambda0MassHypo_pminuspiplus", "Double_t", NULL);
    varLambda0MassHypo_pminuspiplus->FixedMassDaughtersTwoBodyDecayMotherMass(
        _rdcr->GetInterimLeafByName(pminus_px),
        _rdcr->GetInterimLeafByName(pminus_py),
        _rdcr->GetInterimLeafByName(pminus_pz),
        938.272046,
        _rdcr->GetInterimLeafByName(pplus_px),
        _rdcr->GetInterimLeafByName(pplus_py),
        _rdcr->GetInterimLeafByName(pplus_pz),
        139.57018);
    _rdcr->RegisterDoubleLeaf(varLambda0MassHypo_pminuspiplus);
  summary.Add("Veto fit constraints", mass_hypo_constraints);
}
}
