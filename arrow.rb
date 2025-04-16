class Arrow < Formula
    desc "Vector database with persistent storage built in Rust"
    homepage "https://github.com/varadanvk/arrow"
    url "https://github.com/varadanvk/arrow/archive/refs/tags/0.0.1.tar.gz"
    sha256 "860da5d51c99cfada48e902bfa493dda40f2201dcce50cfd4954931a5569165f"
    license "MIT"
  
    depends_on "rust" => :build
  
    def install
      system "cargo", "build", "--release", "--bin", "arrow"
      bin.install "target/release/arrow"
    end
  
    test do
      system "#{bin}/arrow", "--version"
    end
  end
  
  