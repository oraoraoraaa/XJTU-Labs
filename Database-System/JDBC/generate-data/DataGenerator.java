import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

public class DataGenerator {
    private static final String URL = "jdbc:opengauss://192.168.39.160:7654/mydb";
    private static final String USER = "dbremote";
    private static final String PASS = "dbremote:399";

    public static void main(String[] args) throws Exception {
        int students = 1000;
        int courses = 100;
        int enrollments = 20000;
        int threads = 8;
        long runId = System.currentTimeMillis();
        String studentNamesPath = null;
        String teacherNamesPath = null;

        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--students": students = Integer.parseInt(args[++i]); break;
                case "--courses": courses = Integer.parseInt(args[++i]); break;
                case "--enrollments": enrollments = Integer.parseInt(args[++i]); break;
                case "--threads": threads = Integer.parseInt(args[++i]); break;
                case "--student-names": studentNamesPath = args[++i]; break;
                case "--teacher-names": teacherNamesPath = args[++i]; break;
                case "--task2": students=5000; courses=1000; enrollments=200000; break;
            }
        }

        Class.forName("org.opengauss.Driver");

        List<String> studentNames = loadNames(studentNamesPath);
        List<String> teacherNames = loadNames(teacherNamesPath);

        generateStudents(students, threads, runId, studentNames);
        generateCourses(courses, threads, runId, teacherNames);
        generateEnrollments(students, courses, enrollments, threads, runId);
        performDeletionOfLowGrades(200);
        printRowCounts();

        System.out.println("Data generation finished.");
    }

    private static void generateStudents(int n, int threads, long runId, List<String> studentNames) throws InterruptedException {
        ExecutorService es = Executors.newFixedThreadPool(threads);
        java.util.List<Future<?>> futures = new java.util.ArrayList<>();
        int batch = 200;
        for (int start = 1; start <= n; start += batch) {
            int s = start;
            int end = Math.min(n, start + batch - 1);
            futures.add(es.submit(() -> {
                try (Connection conn = DriverManager.getConnection(URL, USER, PASS)) {
                    conn.setAutoCommit(false);
                    String sql = "INSERT INTO \"public\".\"S799\" (\"S_num\",\"SNAME\",\"SEX\",\"BDATE\",\"HEIGHT\",\"DORM\") VALUES (?,?,?,?,?,?)";
                    try (PreparedStatement ps = conn.prepareStatement(sql)) {
                        Random rnd = new Random();
                        for (int i = s; i <= end; i++) {
                            String id = String.format("%013d%04d", runId % 1000000000000L, i);
                            String name = pickName(studentNames, i, "Student" + id);
                            ps.setString(1, id);
                            ps.setString(2, name);
                            ps.setString(3, rnd.nextBoolean() ? "男" : "女");
                            int year = 2000 + rnd.nextInt(7);
                            int month = 1 + rnd.nextInt(12);
                            int day = 1 + rnd.nextInt(28);
                            ps.setDate(4, java.sql.Date.valueOf(String.format("%04d-%02d-%02d", year, month, day)));
                            double h = 1.50 + rnd.nextDouble() * 0.45;
                            ps.setBigDecimal(5, new java.math.BigDecimal(String.format("%.2f", h)));
                            ps.setString(6, "Dorm" + (1 + rnd.nextInt(50)));
                            ps.addBatch();
                        }
                        ps.executeBatch();
                    }
                    conn.commit();
                } catch (SQLException e) {
                    throw new RuntimeException(e);
                }
            }));
        }
        es.shutdown();
        waitForFutures(futures);
        es.awaitTermination(1, TimeUnit.HOURS);
    }

    private static void generateCourses(int n, int threads, long runId, List<String> teacherNames) throws InterruptedException {
        ExecutorService es = Executors.newFixedThreadPool(threads);
        java.util.List<Future<?>> futures = new java.util.ArrayList<>();
        int batch = 200;
        for (int start = 1; start <= n; start += batch) {
            int s = start;
            int end = Math.min(n, start + batch - 1);
            futures.add(es.submit(() -> {
                try (Connection conn = DriverManager.getConnection(URL, USER, PASS)) {
                    conn.setAutoCommit(false);
                    String sql = "INSERT INTO \"public\".\"C799\" (\"C_num\",\"CNAME\",\"PERIOD\",\"CREDIT\",\"TEACHER\") VALUES (?,?,?,?,?)";
                    try (PreparedStatement ps = conn.prepareStatement(sql)) {
                        for (int i = s; i <= end; i++) {
                            String prefix = i <= (n / 2) ? "CS" : "EE";
                            String cnum = String.format("%s-%013d%04d", prefix, runId % 1000000000000L, i);
                            String teacher = pickName(teacherNames, i, "Teacher" + i);
                            ps.setString(1, cnum);
                            ps.setString(2, "Course" + i);
                            ps.setInt(3, 20 + (i % 81));
                            ps.setBigDecimal(4, new java.math.BigDecimal(String.valueOf(1 + (i % 5))));
                            ps.setString(5, teacher);
                            ps.addBatch();
                        }
                        ps.executeBatch();
                    }
                    conn.commit();
                } catch (SQLException e) {
                    throw new RuntimeException(e);
                }
            }));
        }
        es.shutdown();
        waitForFutures(futures);
        es.awaitTermination(1, TimeUnit.HOURS);
    }

    private static void generateEnrollments(int students, int courses, int total, int threads, long runId) throws InterruptedException {
        ExecutorService es = Executors.newFixedThreadPool(threads);
        java.util.List<Future<?>> futures = new java.util.ArrayList<>();
        int batch = 1000;
        int threadCount = Math.max(1, threads);
        for (int start = 0; start < total; start += batch) {
            int s = start;
            int end = Math.min(total, start + batch);
            futures.add(es.submit(() -> {
                try (Connection conn = DriverManager.getConnection(URL, USER, PASS)) {
                    conn.setAutoCommit(false);
                    String sql = "INSERT INTO \"public\".\"SC799\" (\"S_num\",\"C_num\",\"GRADE\") VALUES (?,?,?)";
                    try (PreparedStatement ps = conn.prepareStatement(sql)) {
                        Random rnd = new Random(20260601L + s);
                        for (int i = s; i < end; i++) {
                            long globalIndex = i;
                            int studentNo = (int) (globalIndex % students) + 1;
                            int courseNo = (int) ((globalIndex / students) % courses) + 1;
                            String sid = String.format("%013d%04d", runId % 1000000000000L, studentNo);
                            String prefix = courseNo <= (courses / 2) ? "CS" : "EE";
                            String cid = String.format("%s-%013d%04d", prefix, runId % 1000000000000L, courseNo);
                            if (rnd.nextDouble() < 0.08) {
                                ps.setNull(3, java.sql.Types.DECIMAL);
                            } else {
                                double g = Math.round((rnd.nextDouble() * 100.0) * 10.0) / 10.0;
                                ps.setBigDecimal(3, new java.math.BigDecimal(String.format("%.1f", g)));
                            }
                            ps.setString(1, sid);
                            ps.setString(2, cid);
                            ps.addBatch();
                        }
                        ps.executeBatch();
                    }
                    conn.commit();
                } catch (SQLException e) {
                    throw new RuntimeException(e);
                }
            }));
        }
        es.shutdown();
        waitForFutures(futures);
        es.awaitTermination(2, TimeUnit.HOURS);
    }

    private static void performDeletionOfLowGrades(int toDelete) {
        String sql = "DELETE FROM \"public\".\"SC799\" WHERE ctid IN (SELECT ctid FROM \"public\".\"SC799\" WHERE (\"GRADE\" < 60 OR \"GRADE\" IS NULL) LIMIT ?)";
        try (Connection conn = DriverManager.getConnection(URL, USER, PASS);
             PreparedStatement ps = conn.prepareStatement(sql)) {
            ps.setInt(1, toDelete);
            int deleted = ps.executeUpdate();
            System.out.println("Deleted " + deleted + " low-grade/NULL records from SC799.");
        } catch (SQLException e) {
            throw new RuntimeException(e);
        }
    }

    private static void waitForFutures(java.util.List<Future<?>> futures) {
        for (Future<?> future : futures) {
            try {
                future.get();
            } catch (Exception e) {
                throw new RuntimeException("A worker failed while generating data", e);
            }
        }
    }

    private static void printRowCounts() {
        String sql = "SELECT 'S799' AS tbl, COUNT(*) FROM \"public\".\"S799\" UNION ALL " +
                "SELECT 'C799', COUNT(*) FROM \"public\".\"C799\" UNION ALL " +
                "SELECT 'SC799', COUNT(*) FROM \"public\".\"SC799\"";
        try (Connection conn = DriverManager.getConnection(URL, USER, PASS);
             PreparedStatement ps = conn.prepareStatement(sql);
             ResultSet rs = ps.executeQuery()) {
            System.out.println("Current row counts:");
            while (rs.next()) {
                System.out.println("  " + rs.getString(1) + ": " + rs.getLong(2));
            }
        } catch (SQLException e) {
            throw new RuntimeException(e);
        }
    }

    private static List<String> loadNames(String filePath) {
        if (filePath == null || filePath.isBlank()) {
            return new ArrayList<>();
        }
        try {
            Path path = Paths.get(filePath);
            if (!Files.exists(path)) {
                return new ArrayList<>();
            }
            List<String> names = new ArrayList<>();
            for (String line : Files.readAllLines(path)) {
                String trimmed = line.trim();
                if (!trimmed.isEmpty()) {
                    names.add(trimmed);
                }
            }
            return names;
        } catch (IOException e) {
            throw new RuntimeException("Failed to read names from: " + filePath, e);
        }
    }

    private static String pickName(List<String> names, int index, String fallback) {
        if (names == null || names.isEmpty()) {
            return fallback;
        }
        return names.get(Math.floorMod(index - 1, names.size()));
    }
}
