class Arrow < Formula
    desc "Vector database with persistent storage built in Rust!"
    homepage "https://github.com/varadanvk/arrow"
    url "https://github.com/varadanvk/arrow/archive/refs/tags/0.0.2.tar.gz"
    sha256 "7a100495e92d575afb1954b01899c7252a9719f5323e7e7d945b1ef4a831a8d9"
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
  
  