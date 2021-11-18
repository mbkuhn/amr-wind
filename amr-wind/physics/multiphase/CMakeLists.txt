target_sources(${amr_wind_lib_name}
  PRIVATE
  MultiPhase.cpp
  VortexPatch.cpp
  VortexPatchScalarVel.cpp
  ZalesakDisk.cpp
  ZalesakDiskScalarVel.cpp
  DamBreak.cpp
  SloshingTank.cpp
  BreakingWaves.cpp
  RainDrop.cpp
  )

<<<<<<< HEAD
add_subdirectory(wave_basin)
=======
add_subdirectory(numerical_wave_tank)

>>>>>>> added a basic numerical wave tank capability with only linear waves at the moment
