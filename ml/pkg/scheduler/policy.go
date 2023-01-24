package scheduler

import (
	"sync"

	"github.com/nwangfw/kubeml/ml/pkg/api"
	"go.uber.org/zap"
)

const (
	SpeedScaleDownThreshold = 1
	SpeedScaleUpThreshold   = 1
)

// SchedulerPolicy defines the methods needed to be implemented by the scheduler
// in order to support the tasks from KubeML, these involve how to calculate the parallelism
// of the task based on previous performance of the task
type (
	SchedulerPolicy interface {
		// calculate paralellism returns the parallelism for the next epoch
		calculateParallelism(task api.TrainTask) (parallelism int, op TaskOperation)
		taskFinished(taskId string)
	}

	SpeedBasedPolicy struct {
		logger *zap.Logger

		// timeCache saves the throughput from previous epochs
		// of the different jobs and is used to reactively scale up or down
		// the parallelism when we see that the time elapsed for an epoch
		// increases or decreases
		timeCache           map[string]float64
		timeBestCache       map[string]float64
		timeBestParallelism map[string]int

		mu *sync.RWMutex
	}
)

func makeSpeedPolicy(logger *zap.Logger) SpeedBasedPolicy {
	return SpeedBasedPolicy{
		logger:              logger.Named("speed-policy"),
		timeCache:           make(map[string]float64),
		timeBestCache:       make(map[string]float64),
		timeBestParallelism: make(map[string]int),
		mu:                  &sync.RWMutex{},
	}
}

// calculateParallelism for the throughput based policy simply scales up if the performance
// is better or slightly worse than in previous epochs (given by the scale-up threshold), and scales
// down if the performance is much worse.
//
// In between those thresholds the parallelism is kept untouched
func (sp SpeedBasedPolicy) calculateParallelism(task api.TrainTask) (parallelism int, op TaskOperation) {

	sp.mu.RLock()
	prevTime, exists := sp.timeCache[task.Job.JobId]
	sp.mu.RUnlock()

	// If it is the first epoch and we do not have a history
	// of this task, simply return the debug parallelism
	if !exists {
		sp.mu.Lock()
		sp.timeCache[task.Job.JobId] = 0
		sp.mu.Unlock()

		return task.Parameters.Options.DefaultParallelism, CreateTask

	} else {

		switch {
		case prevTime == 0:
			sp.logger.Debug("No previous time, increasing parallelism")
			sp.timeCache[task.Job.JobId] = task.Job.State.ElapsedTime
			sp.timeBestCache[task.Job.JobId] = task.Job.State.ElapsedTime
			sp.timeBestParallelism[task.Job.JobId] = task.Job.State.Parallelism
			return task.Job.State.Parallelism + 2, UpdateTask

		// If the new time is better than the prevTime
		// always scale up and set a new reference time
		case task.Job.State.ElapsedTime-prevTime <= SpeedScaleUpThreshold*sp.timeBestCache[task.Job.JobId]:
			sp.logger.Debug("Time is better, scaling up")
			sp.timeCache[task.Job.JobId] = task.Job.State.ElapsedTime
			sp.timeBestCache[task.Job.JobId] = task.Job.State.ElapsedTime - prevTime
			sp.timeBestParallelism[task.Job.JobId] = task.Job.State.Parallelism
			return task.Job.State.Parallelism + 2, UpdateTask

		// If the performance is much worse (20%) than the reference
		// time, downscale and set a new reference time
		case task.Job.State.ElapsedTime-prevTime >= SpeedScaleDownThreshold*sp.timeBestCache[task.Job.JobId]:
			sp.logger.Debug("Time is worse, scaling down")
			sp.timeCache[task.Job.JobId] = task.Job.State.ElapsedTime
			return sp.timeBestParallelism[task.Job.JobId], UpdateTask

		default:
			sp.logger.Debug("Time is worse within the limits, keeping parallelism")
			return task.Job.State.Parallelism, UpdateTask
		}

	}

}

// taskFinished handles the finish of the task, here simply deletes it from
// the time cache
func (sp SpeedBasedPolicy) taskFinished(taskId string) {
	sp.mu.Lock()
	defer sp.mu.Unlock()
	delete(sp.timeCache, taskId)
	delete(sp.timeBestCache, taskId)
	delete(sp.timeBestParallelism, taskId)
}
