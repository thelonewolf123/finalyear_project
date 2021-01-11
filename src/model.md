# DB design - client

## Visiter Table 
	- Id
	- Photo

## CameraMap Table
	- Id
	- Location
	- Camera Number	

## DataMap Table
	- Id
	- Visiter Id (Many to many)
	- CameraMap Id (Many to many)
	- Time


