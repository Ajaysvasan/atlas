package com.fyp.atlas.service;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.text.Normalizer;
import java.util.HashSet;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.regex.Pattern;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

import com.fyp.atlas.config.AtlasProperties;

import org.springframework.stereotype.Component;
import org.springframework.web.multipart.MultipartFile;

/**
 * First-line checks on an uploaded file, so obviously bad files never cross to Python. Python repeats
 * every check and stays authoritative. Content is read as a stream, never executed or unpacked to disk.
 */
@Component
public class UploadValidator {

	public record Rejection(String code, String message) {
	}

	static final int MAX_NAME_LENGTH = 200;
	static final int TEXT_PROBE_BYTES = 64 * 1024;
	private static final Pattern UNSAFE = Pattern.compile("[\\x00-\\x1f\\x7f/\\\\:*?\"<>|]");
	private static final Map<String, String> OFFICE = Map.of("docx", "word/", "pptx", "ppt/", "xlsx", "xl/");

	private final AtlasProperties.Upload settings;

	public UploadValidator(AtlasProperties properties) {
		this.settings = properties.upload();
	}

	/** A display name with no directory part, control characters or path syntax; never empty. */
	public static String safeFilename(String original) {
		String name = Normalizer.normalize(original == null ? "" : original, Normalizer.Form.NFC);
		int slash = Math.max(name.lastIndexOf('/'), name.lastIndexOf('\\'));
		name = UNSAFE.matcher(name.substring(slash + 1)).replaceAll("_").strip();
		while (name.endsWith(".") || name.startsWith(".")) {
			name = name.endsWith(".") ? name.substring(0, name.length() - 1) : name.substring(1);
		}
		if (name.length() > MAX_NAME_LENGTH) {
			String extension = extension(name);
			name = extension.isEmpty() || extension.length() > 10 ? name.substring(0, MAX_NAME_LENGTH)
					: name.substring(0, MAX_NAME_LENGTH - extension.length() - 1) + "." + extension;
		}
		return name.isBlank() ? "upload" : name;
	}

	public static String extension(String name) {
		int dot = name.lastIndexOf('.');
		return dot < 0 || dot == name.length() - 1 ? "" : name.substring(dot + 1).toLowerCase(Locale.ROOT);
	}

	public Optional<Rejection> validate(MultipartFile file, String safeName) {
		String kind = extension(safeName);
		if (!settings.allowedExtensions().contains(kind)) {
			return reject("UNSUPPORTED_TYPE", kind.isEmpty() ? "Files need an extension."
					: "'." + kind + "' files are not accepted.");
		}
		if (file.isEmpty()) {
			return reject("EMPTY", "The file is empty.");
		}
		if (file.getSize() > settings.maxFileSize().toBytes()) {
			return reject("TOO_LARGE", "The file is larger than " + settings.maxFileSize() + ".");
		}
		try {
			if (kind.equals("pdf")) {
				return pdf(file);
			}
			if (OFFICE.containsKey(kind)) {
				return office(file, OFFICE.get(kind));
			}
			return text(file);
		}
		catch (IOException error) {
			return reject("MALFORMED", "The file could not be read.");
		}
	}

	private Optional<Rejection> pdf(MultipartFile file) throws IOException {
		try (InputStream in = file.getInputStream()) {
			String head = new String(in.readNBytes(1024), StandardCharsets.ISO_8859_1);
			return head.contains("%PDF-") ? Optional.empty() : reject("MALFORMED", "The file is not a PDF.");
		}
	}

	/** Every entry's bytes are counted as they decompress: declared sizes in a zip can lie. */
	private Optional<Rejection> office(MultipartFile file, String folder) throws IOException {
		long limit = settings.maxUncompressedArchiveSize().toBytes();
		long total = 0;
		int entries = 0;
		Set<String> names = new HashSet<>();
		byte[] buffer = new byte[64 * 1024];
		try (ZipInputStream zip = new ZipInputStream(file.getInputStream())) {
			ZipEntry entry;
			while ((entry = zip.getNextEntry()) != null) {
				if (++entries > settings.maxArchiveEntries()) {
					return reject("MALFORMED", "The document has too many parts.");
				}
				String name = entry.getName();
				if (name.startsWith("/") || name.contains("..")) {
					return reject("MALFORMED", "The document contains an unsafe part name.");
				}
				names.add(name);
				int read;
				while ((read = zip.read(buffer)) > 0) {
					total += read;
					if (total > limit) {
						return reject("TOO_LARGE", "The document expands beyond the allowed size.");
					}
				}
			}
		}
		catch (IllegalArgumentException | java.util.zip.ZipException error) {
			return reject("MALFORMED", "The document could not be read.");
		}
		if (entries == 0 || !names.contains("[Content_Types].xml")
				|| names.stream().noneMatch(n -> n.startsWith(folder))) {
			return reject("MALFORMED", "The file is not the Office document its extension claims.");
		}
		if (total > 1024 * 1024 && total / Math.max(1, file.getSize()) > settings.maxCompressionRatio()) {
			return reject("MALFORMED", "The document is compressed suspiciously far.");
		}
		return Optional.empty();
	}

	private Optional<Rejection> text(MultipartFile file) throws IOException {
		byte[] probe;
		try (InputStream in = file.getInputStream()) {
			probe = in.readNBytes(TEXT_PROBE_BYTES);
		}
		int controls = 0;
		for (byte b : probe) {
			if (b == 0) {
				return reject("MALFORMED", "The file is binary, not text.");
			}
			if ((b & 0xff) < 32 && b != 9 && b != 10 && b != 12 && b != 13) {
				controls++;
			}
		}
		if (probe.length > 0 && controls * 100 > probe.length) {
			return reject("MALFORMED", "The file is binary, not text.");
		}
		return Optional.empty();
	}

	private static Optional<Rejection> reject(String code, String message) {
		return Optional.of(new Rejection(code, message));
	}
}
