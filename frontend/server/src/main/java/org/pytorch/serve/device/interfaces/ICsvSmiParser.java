package org.pytorch.serve.device.interfaces;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.function.Function;
import org.pytorch.serve.device.Accelerator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public interface ICsvSmiParser {
    static final Logger csvSmiParserLogger = LoggerFactory.getLogger(ICsvSmiParser.class);

    /**
     * Parses CSV output from SMI commands and converts it into a list of Accelerator objects.
     *
     * @param csvOutput The CSV string output from an SMI command.
     * @param parsedAcceleratorIds A set of accelerator IDs to consider. If empty, all accelerators
     *     are included.
     * @param parseFunction A function that takes an array of CSV fields and returns an Accelerator
     *     object. This function should handle the specific parsing logic for different SMI command
     *     outputs.
     * @return An ArrayList of Accelerator objects parsed from the CSV output.
     * @throws NumberFormatException If there's an error parsing numeric fields in the CSV.
     *     <p>This method provides a general way to parse CSV output from various SMI commands. It
     *     skips the header line of the CSV, then applies the provided parseFunction to each
     *     subsequent line. Accelerators are only included if their ID is in parsedAcceleratorIds,
     *     or if parsedAcceleratorIds is empty (indicating all accelerators should be included).
     *     <p>The parseFunction parameter allows for flexibility in handling different CSV formats
     *     from various SMI commands. This function should handle the specific logic for creating an
     *     Accelerator object from a line of CSV data.
     */
    default ArrayList<Accelerator> csvSmiOutputToAccelerators(
            final String csvOutput,
            final LinkedHashSet<Integer> parsedGpuIds,
            Function<String[], Accelerator> parseFunction) {
        final ArrayList<Accelerator> accelerators = new ArrayList<>();

        List<String> lines = Arrays.asList(csvOutput.split("\n"));

        final boolean addAll = parsedGpuIds.isEmpty();

        lines.stream()
                .skip(1) // Skip the header line
                .forEach(
                        line -> {
                            final String[] parts = line.split(",");
                            try {
                                Accelerator accelerator = parseFunction.apply(parts);
                                if (accelerator != null
                                        && (addAll
                                                || parsedGpuIds.contains(
                                                        accelerator.getAcceleratorId()))) {
                                    accelerators.add(accelerator);
                                }
                            } catch (final NumberFormatException e) {
                                csvSmiParserLogger.warn(
                                        "Failed to parse GPU ID: " + parts[1].trim(), e);
                            }
                        });

        return accelerators;
    }
}
