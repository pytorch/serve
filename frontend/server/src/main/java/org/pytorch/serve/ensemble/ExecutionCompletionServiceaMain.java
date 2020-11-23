package org.pytorch.serve.ensemble;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.CompletionService;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorCompletionService;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

class MultiplyingTask implements Callable {
    int a;
    int b;
    long sleepTime;
    String taskName;

    public MultiplyingTask(String taskName, int a, int b, long sleepTime) {
        this.taskName = taskName;
        this.a = a;
        this.b = b;
        this.sleepTime = sleepTime;
    }

    @Override
    public Integer call() throws Exception {
        System.out.println("Started taskName: " + taskName);
        int result = a * b;
        Thread.sleep(sleepTime);
        System.out.println("Completed taskName: " + taskName);
        return result;
    }
}

public class ExecutionCompletionServiceaMain {

    public static void main(String[] args) {
        MultiplyingTask multiplyingTask1 = new MultiplyingTask("Task 1", 10, 20, 6000l);
        MultiplyingTask multiplyingTask2 = new MultiplyingTask("Task 2", 30, 40, 4000l);
        MultiplyingTask multiplyingTask3 = new MultiplyingTask("Task 3", 40, 50, 3000l);
        MultiplyingTask multiplyingTask4 = new MultiplyingTask("Task 4", 50, 60, 1000l);

        ExecutorService executorService = Executors.newFixedThreadPool(10);
        CompletionService<Integer> executorCompletionService =
                new ExecutorCompletionService<>(executorService);
        List<Future<Integer>> futures = new ArrayList<Future<Integer>>();
        futures.add(executorCompletionService.submit(multiplyingTask1));
        futures.add(executorCompletionService.submit(multiplyingTask2));
        futures.add(executorCompletionService.submit(multiplyingTask3));
        futures.add(executorCompletionService.submit(multiplyingTask4));

        Integer jobs = 5;
        while (jobs > 0) {
            try {
                Integer result = executorCompletionService.take().get();
                futures.add(executorCompletionService.submit(multiplyingTask4));
                futures.add(executorCompletionService.submit(multiplyingTask4));
                jobs--;
                System.out.println("Result: " + result);
                if (result == 1200) {
                    futures.add(executorCompletionService.submit(multiplyingTask4));
                }
            } catch (InterruptedException e) {
                e.printStackTrace();
            } catch (ExecutionException e) {
                e.printStackTrace();
            }
        }
        executorService.shutdown();
    }
}
