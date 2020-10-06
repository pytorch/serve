import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import org.apache.log4j.spi.LoggingEvent;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;



public class CachingSingletonAppender {

    private static final CachingSingletonAppender theInstance = new CachingSingletonAppender();
    private List<LogEventListener> listeners;
    private List<LogEvent> eventCache;

    private CachingSingletonAppender() {
        listeners = new ArrayList<>();
        eventCache = new ArrayList<>();
    }

    public static CachingSingletonAppender getInstance() {
        return theInstance;
    }

    public void append(LoggingEvent le) {
        LogEvent event = new LogEvent(le.getLevel().toString(), le.getMessage().toString(), new Date(le.getTimeStamp()));
        System.out.println("**********");
        System.out.print(event.getMessage());
        if (!listeners.isEmpty()) {
            for (LogEventListener listener : listeners) {
                listener.handle(event);
            }
        } else {
            eventCache.add(event);
        }
    }

    public void addLoggingEventListener(LogEventListener listener) {
        listeners.add(listener);
        if (listeners.size() == 1 && !eventCache.isEmpty()) {
            sendAndClearCache();
        }
    }

    public void removeLoggingEventListener(LogEventListener listener) {
        listeners.remove(listener);
    }

    synchronized private void sendAndClearCache() {
        for (LogEventListener listener : listeners) {
            for (LogEvent event : eventCache) {
                listener.handle(event);
                System.out.println("**********");
                System.out.print(event.getMessage());
            }
        }
        eventCache.clear();
    }
}