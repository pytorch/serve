package org.pytorch.serve.job;

import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.TimeUnit;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class JobGroup {
    private static final Logger logger = LoggerFactory.getLogger(JobGroup.class);
    String groupId;
    LinkedBlockingDeque<Job> jobs;
    int maxJobQueueSize;
    boolean finished;

    public JobGroup(String groupId, int maxJobQueueSize) {
        this.groupId = groupId;
        this.maxJobQueueSize = maxJobQueueSize;
        this.jobs = new LinkedBlockingDeque<>(maxJobQueueSize);
        this.finished = false;
    }

    public boolean appendJob(Job job) {
        return jobs.offer(job);
    }

    public Job pollJob(long timeout) {
        if (finished) {
            return null;
        }
        try {
            return jobs.poll(timeout, TimeUnit.MILLISECONDS);
        } catch (InterruptedException e) {
            logger.error("Failed to poll a job from group {}", groupId, e);
        }
        return null;
    }

    public String getGroupId() {
        return groupId;
    }

    public void setFinished(boolean sequenceEnd) {
        this.finished = sequenceEnd;
    }

    public boolean isFinished() {
        return this.finished;
    }
}
