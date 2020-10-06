import org.apache.log4j.AppenderSkeleton;
import org.apache.log4j.spi.LoggingEvent;

public class InternalAppenderWrapper extends AppenderSkeleton {

    private CachingSingletonAppender appender = CachingSingletonAppender.getInstance();

    @Override
    protected void append(LoggingEvent le) {
        appender.append(le);
    }

    @Override
    public void close() {
    }

    @Override
    public boolean requiresLayout() {
        return false;
    }

}