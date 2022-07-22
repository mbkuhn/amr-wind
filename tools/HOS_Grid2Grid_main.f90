!! Program Start ----------------------------------------------------
	Program Main
	!! ------------------------------------------------------------------
	use modCommG2G			!! Use Communication Module
	!! ------------------------------------------------------------------
	Implicit None
	!! Variables --------------------------------------------------------
	Integer,Parameter      :: nChar = 300			!! Default Character Length
	Character(len = nChar) :: grid2gridPath		!! libGrid2Grid.so Path

	integer                :: hosIndex				!! HOS Index
	Character(len = nChar) :: hosSolver				!! HOS Solver (Ocean or NWT)
	Character(len = nChar) :: hosFileName			!! HOS Result File Path

	Character(len = nChar) :: dictFilePath   		!! HOS Solver (Ocean or NWT)
	Character(len = nChar) :: OutFilePath, istr

	Double precision       :: zMin, zMax			!! Surf2Vol Domain
	integer                :: nZmin, nZmax		!! Number of vertical grid
	Double precision       :: zMinRatio, zMaxRatio	!! Grading ratio (=3)

	Double precision       :: t, dt			!! Simulation Time, dt
	Double precision       :: x, y, z		!! Computation Point
	Double precision       :: eta, u, v, w
	!! Dummy variables --------------------------------------------------
	integer                :: it				!! Dummy time loop integer
	integer                :: i,j,k
	integer                :: nx,ny,nz
	Double precision       :: Lx,Ly

	!! Program Body -----------------------------------------------------

	!!!... Write Program Start
	write(*,*) "Test program (Connect to Fortran) to use Grid2Grid shared library"

	!!!... Set libGrid2Grid.so path.
	!!!    It is recommended to use absolute path
	! grid2gridPath = "/usr/lib/libGrid2Grid.so"	(if soft link is made)
	grid2gridPath = "/Users/mkuhn/Grid2Grid/lib/libGrid2Grid.so"

	!!!... Load libGrid2Grid.so and connect subroutines
	Call callGrid2Grid(grid2gridPath)

	!!!... Declare HOS Index
	hosIndex = -1

	!!!... Set HOS Type (Ocean or NWT)
	hosSolver = "Ocean"

	!!!... Set HOS Result file Path
	hosFileName = "../Results/modes_HOS_SWENSE.dat"
	!hosFileName = "../modes_HOS_SWENSE.hdf5"

	!dictFilePath = "Grid2Grid.dict"

	!!!... Set HOS Surf2Vol Domain and Vertical Grid
	zMin = -35.d0; 				zMax =  10.d0
	nZmin = 45; 					nZmax = 18
	zMinRatio = 1.d0; 		zMaxRatio = 1.d0

	!!... Initialize Grid2Grid and Get HOS Index
	Call initializeGrid2Grid(hosSolver, hosFileName, zMin, zMax, nZmin, nZmax, zMinRatio, zMaxRatio, hosIndex)
	!Call initializeGrid2GridDict(dictFilePath, hosIndex)

	!! Time Information
	t  = 0.0d0; 		dt = 0.5d0

	!! 2D mesh info
	nx = 64; ny = 64
	Lx = 2.8499216855720260E+03; Ly = 2.8499216855720260E+03;

	!! Modified mesh in z
	nz = nZmin + nZmax + 1

	!! Time Loop
	do it = 0,100*2
		!! Make filename
		write(istr,"(i10)") it
		write(OutFilePath,"(a12,a,a4)") "HOSGridData_", trim(adjustl(istr)), ".txt"
		!! Correct HOS Vol2VOl for given time
		Call correctGrid2Grid(hosIndex, t)
		!! Open new file dedicated to this timestep
		open(1,file=OutFilePath)
		write(1,"(a, f15.8)")  "HOS Time = ", t
		write(1,"(a, f15.8)")  "HOS dt = ", dt
		write(1,"(a, I5, a, f15.8, a, f15.8)")  "nx = ", nx, ", Lx = ", Ly
		write(1,"(a, I5, a, f15.8, a, f15.8)")  "ny = ", nx, ", Ly = ", Ly
		write(1,"(a, I5, a, f15.8, a, f15.8)")  "nz = ", nz, ", zmin = ", zMin, ", zmax = ", zMax
                write(1,"(a)") "eta at i,j followed by u,v,w at same i,j with k ascending"
  
		!! 2D grid
		do i = 1,nx
			do j = 1,ny
				!! 2D location
				x = (i-1)*(Lx/nx); y = (j-1)*(Ly/ny)

				!! Get Wave Elevation
				Call getHOSeta(hosIndex, x, y , t, eta)
				!! Record Wave Elevation
				write(1,"(f15.8)")  eta

				!! Go through points in z
				do k = 1, nz
					!! Height location
					z = zMin + (zMax-zMin)/nz * (k-1)
					!! Get Flow Velocity
					Call getHOSU(hosIndex, x, y, z, t, u, v ,w)

					!! Write Flow Information
					write(1,"(3f15.8)")  u,v,w
				end do
			end do
		end do

		!! Close current file
		close(1)
		!! Time Update
		t = t + dt
	enddo

	!! Write End of Program
	write(*,*) "Test program (Connect to Fortran) is done ..."
	!! ------------------------------------------------------------------
	End Program
	!! ------------------------------------------------------------------
