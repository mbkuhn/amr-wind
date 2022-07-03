#ifndef OCEAN_WAVES_OPS_H
#define OCEAN_WAVES_OPS_H

#include "amr-wind/ocean_waves/OceanWavesTypes.H"
#include "AMReX_Vector.H"
#include "amr-wind/CFDSim.H"

namespace amr_wind {
namespace ocean_waves {
namespace ops {

/** Read user inputs from an input file for a given immersed boundary object.
 *
 *  \ingroup ocean waves
 *
 *  This operator is called within Physics::pre_init_actions() and, therefore,
 *  does not have access to mesh data. Actions that require mesh information for
 *  initialization should be deferred until ops::init_data_structures call.
 */
template <typename WaveTheoryTrait, typename = void>
struct ReadInputsOp;

/** Perform one-time initialization of the immersed boundary object.
 *
 *  \ingroup ocean waves
 *
 *  This operator is responsible for allocating memory ensuring that the
 *  ocean waves instance has been initialized and ready to be used by the
 * ocean waves physics instance. This function is called once during
 * IB::post_init_actions
 *
 */
template <typename WaveTheoryTrait, typename = void>
struct InitDataOp;

/** Update the positions of the immersed boundary nodes.
 *
 *  \ingroup ocean waves
 *
 *  After this call, the interface expects updated position vectors in the
 *  vel_pos array of the ocean waves grid instance. These positions will
 * be used the sample the velocity field during the simulation.
 */
template <typename WaveTheoryTrait, typename = void>
struct UpdateFSOp;

/** Process updated velocities from CFD flow field.
 *
 *  \ingroup ocean waves
 *
 *  The immersed boundary physics interface will populate the `vel` array of the
 *  ocean waves grid instance before calling this function. The updated
 *  velocities can be further processed by specializing this function.
 */
template <typename WaveTheoryTrait, typename = void>
struct UpdateVelOp;

template <typename WaveTheoryTrait, typename = void>
struct ProcessOutputsOp;

} // namespace ops
} // namespace ocean_waves
} // namespace amr_wind

#endif /* OCEAN_WAVES_OPS_H */
