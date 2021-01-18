package api

const (

	// Constants to save and retrieve the gradients
	WeightSuffix   = ".weight"
	BiasSuffix     = ".bias"
	GradientSuffix = ".grad"
)

// Addresses and ports of services
const (
	// Address to access the fission router
	STORAGE_ADDRESS = "http://storagesvc"
	ROUTER_ADDRESS = "http://router.fission"
	MONGO_ADDRESS  = "mongo.default"
	MONGO_PORT     = 27017
	REDIS_ADDRESS  = "redis.default"
)

// Debug
const (
	MONGO_ADDRESS_DEBUG = "mongodb://192.168.99.101:30933"
	STORAGE_ADDRESS_DEBUG = "http://192.168.99.102:80"
	ROUTER_ADDRESS_DEBUG = "http://192.168.99.101:32422"
	REDIS_ADDRESS_DEBUG  = "192.168.99.101"
	REDIS_PORT_DEBUG     = 31618
	DEBUG_PARALLELISM    = 2
	SCHEDULER_DEBUG_PORT = 10200
	PS_DEBUG_PORT = 10300
	CONTROLLER_DEBUG_PORT = 10100
	DEBUG_URL = "http://localhost"
)



