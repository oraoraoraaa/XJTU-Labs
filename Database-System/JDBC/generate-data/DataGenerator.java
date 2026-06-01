import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
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

        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--students": students = Integer.parseInt(args[++i]); break;
                case "--courses": courses = Integer.parseInt(args[++i]); break;
                case "--enrollments": enrollments = Integer.parseInt(args[++i]); break;
                case "--threads": threads = Integer.parseInt(args[++i]); break;
                case "--task2": students=5000; courses=1000; enrollments=200000; break;
            }
        }

        Class.forName("org.opengauss.Driver");

        generateStudents(students, threads);
        generateCourses(courses, threads);
        generateEnrollments(students, courses, enrollments, threads);
        performDeletionOfLowGrades(200);

        System.out.println("Data generation finished.");
    }

    private static void generateStudents(int n, int threads) throws InterruptedException {
        ExecutorService es = Executors.newFixedThreadPool(threads);
        int batch = 200;
        for (int start = 1; start <= n; start += batch) {
            int s = start;
            int end = Math.min(n, start + batch - 1);
            es.submit(() -> {
                try (Connection conn = DriverManager.getConnection(URL, USER, PASS)) {
                    conn.setAutoCommit(false);
                    String sql = "INSERT INTO \"public\".\"S799\" (\"S_num\",\"SNAME\",\"SEX\",\"BDATE\",\"HEIGHT\",\"DORM\") VALUES (?,?,?,?,?,?) ON CONFLICT (\"S_num\") DO NOTHING";
                    try (PreparedStatement ps = conn.prepareStatement(sql)) {
                        Random rnd = new Random();
                        for (int i = s; i <= end; i++) {
                            String id = String.format("%08d", i);
                            ps.setString(1, id);
                            ps.setString(2, "Student" + id);
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
            });
        }
        es.shutdown();
        es.awaitTermination(1, TimeUnit.HOURS);
    }

    private static void generateCourses(int n, int threads) throws InterruptedException {
        ExecutorService es = Executors.newFixedThreadPool(threads);
        int batch = 200;
        for (int start = 1; start <= n; start += batch) {
            int s = start;
            int end = Math.min(n, start + batch - 1);
            es.submit(() -> {
                try (Connection conn = DriverManager.getConnection(URL, USER, PASS)) {
                    conn.setAutoCommit(false);
                    String sql = "INSERT INTO \"public\".\"C799\" (\"C_num\",\"CNAME\",\"PERIOD\",\"CREDIT\",\"TEACHER\") VALUES (?,?,?,?,?) ON CONFLICT (\"C_num\") DO NOTHING";
                    try (PreparedStatement ps = conn.prepareStatement(sql)) {
                        Random rnd = new Random();
                        for (int i = s; i <= end; i++) {
                            String id = String.format("C%05d", i);
                            String prefix = (rnd.nextBoolean() ? "CS" : "EE");
                            String cnum = prefix + "-" + id;
                            ps.setString(1, cnum);
                            ps.setString(2, "Course" + id);
                            ps.setInt(3, 20 + rnd.nextInt(81));
                            ps.setBigDecimal(4, new java.math.BigDecimal(1 + rnd.nextInt(5)));
                            ps.setString(5, "Teacher" + id);
                            ps.addBatch();
                        }
                        ps.executeBatch();
                    }
                    conn.commit();
                } catch (SQLException e) {
                    throw new RuntimeException(e);
                }
            });
        }
        es.shutdown();
        es.awaitTermination(1, TimeUnit.HOURS);
    }

    private static void generateEnrollments(int students, int courses, int total, int threads) throws InterruptedException {
        ExecutorService es = Executors.newFixedThreadPool(threads);
        int batch = 1000;
        Random rnd = new Random();
        for (int start = 0; start < total; start += batch) {
            int s = start;
            int end = Math.min(total, start + batch);
            es.submit(() -> {
                try (Connection conn = DriverManager.getConnection(URL, USER, PASS)) {
                    conn.setAutoCommit(false);
                    String sql = "INSERT INTO \"public\".\"SC799\" (\"S_num\",\"C_num\",\"GRADE\") VALUES (?,?,?) ON CONFLICT (\"S_num\",\"C_num\") DO NOTHING";
                    try (PreparedStatement ps = conn.prepareStatement(sql)) {
                        for (int i = s; i < end; i++) {
                            String sid = String.format("%08d", 1 + rnd.nextInt(students));
                            String cid = (rnd.nextBoolean() ? "CS-" : "EE-") + String.format("C%05d", 1 + rnd.nextInt(courses));
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
            });
        }
        es.shutdown();
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
}
